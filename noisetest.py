"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import os
import time
import copy
import numpy as np
from tqdm import tqdm
import traceback
import logging
from logging import handlers

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True

from torch import optim
from torch.utils.data import SubsetRandomSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils.configs import config
from utils.metric import IOUMetric
from utils.utils import *

from model.SegModel import SegModel, test_model, SegTransformer, SegSwinTransformer, build_model

from dataset.data_loader import beam_dataloader, beam_gendataloader

import cv2

import matplotlib.pyplot as plt
from matplotlib import ticker

def test_net(net, opt, dataset, paths):
    device = opt.device
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    n_classes=opt.n_classes
    net.eval()
    test_step = 0
    iou=IOUMetric(n_classes)
    gt_path, input_path, xx_path, yy_path, xy_path, color_path, pred_path = paths
    for image, label in tqdm(test_loader):
        if opt.datasetindex==1:
            image = image[:,:,3:-3,4:-4]
            label = label[:,:,3:-3,4:-4]
        test_input = deformation_to_img(image)
        label_color = label2color(label, n_classes, opt.color_array)
        #print(label_color.shape)
        
        image = image.to(device=opt.device, dtype=torch.float32)
        label = label.to(device=opt.device, dtype=torch.float32)
        
        pred = net(image)
        pred_color = label2color(pred, n_classes, opt.color_array)
        
        filename = '{:>5d}.png'.format(int(test_step))
        
        gt_file = os.path.join(gt_path, filename)
        #in_file = os.path.join(opt.input_image_path, filename)
        xx_file = os.path.join(xx_path, filename)
        yy_file = os.path.join(yy_path, filename)
        xy_file = os.path.join(xy_path, filename)
        col_file = os.path.join(color_path, filename)
        pred_file = os.path.join(pred_path, filename)
        
        label_color = label_color.transpose((1,2,0))
        test_input = test_input.transpose((1,2,0))
        pred_color = pred_color.transpose((1,2,0))
        
        h,w,c = pred_color.shape
        pred_color = cv2.resize(pred_color, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
        test_input = cv2.resize(test_input, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
        label_color = cv2.resize(label_color, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(gt_file, label_color)

        cv2.imwrite(xx_file, test_input[:,:,0])
        cv2.imwrite(yy_file, test_input[:,:,1])
        cv2.imwrite(xy_file, test_input[:,:,2])
        cv2.imwrite(col_file, test_input)
        cv2.imwrite(pred_file, pred_color)
        
        test_step+=1
        
        pred=pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        if n_classes==1:
            pred[pred>=0.5]=1
            pred[pred<0.5]=0
            label[label>=0.5]=1
            label[label<0.5]=0
            #n,h,w = label.shape
            #label = label.reshape(n,1,h,w)
        else:
            pred= np.argmax(pred,axis=1)
            label = np.argmax(label,axis=1)
        pred = pred.astype(int)
        label = label.astype(int)
        iou.add_batch(pred,label)
    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
    
    print('acc=', acc)
    print('acc_cls=', acc_cls)
    print('iu=', iu)
    print('mean_iu=', mean_iu)
    
    return acc, acc_cls, iu, mean_iu
        
class noise_test_config(object):
    def __init__(self, opt):
        super(noise_test_config, self).__init__()
        self.task = opt.task
        self.datasetindex = opt.datasetindex
        self.datasetname = opt.datasetnames[opt.datasetindex]
        self.save_model_path = './user_data/EN/model_data'
        self.val_path = opt.val_path
        self.n_classes = opt.n_classes
        self.input_channel = opt.input_channel
        self.device = opt.device
        self.color_array = opt.color_array
        # our model configs
        self.plate_swin_model_configs =[opt.Swin64Tconfig, 
                                        opt.Swin64Sconfig, 
                                        opt.Swin64Mconfig, 
                                        opt.Swin64Bconfig, 
                                        opt.Swin64Lconfig]
        self.sleeperbeam_swin_model_configs = [opt.Swin264Tconfig,
                                               opt.Swin264Sconfig,
                                               opt.Swin264Mconfig,
                                               opt.Swin264Bconfig,
                                               opt.Swin264Lconfig]
        
        # model list
        self.model_names, self.model_paths, self.model_configs, self.model_label = self.get_model_list()
        self.noise_lv = [0, 0.04, 0.08, 0.12, 0.16, 0.2]
        
        # noise test result path
        self.noise_result_path = './noise_test_result'
        self.noise_result_path = os.path.join(self.noise_result_path, self.datasetname)
        if not os.path.exists(self.noise_result_path):
            os.makedirs(self.noise_result_path)
        
    def get_model_list(self):
        if self.datasetname == 'plate':
            mymodels = ['Swin64T', 'Swin64S','Swin64M','Swin64B','Swin64L']
            mymodelconfigs = self.plate_swin_model_configs
        elif self.datasetname == 'sleeper_beam':
            mymodels = ['Swin264T', 'Swin264S', 'Swin264M', 'Swin264B', 'Swin264L']
            mymodelconfigs = self.sleeperbeam_swin_model_configs
        else:
            ValueError('datasetname must be plate or sleeper_beam')
            
        model_zoo = ['Unet', 'PSPnet', 'DeepLabV3']
        backbone = ['resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152']
        
        model_names = []
        model_paths =[]
        model_configs = []
        model_label = []
        for model in model_zoo:
            for bone in backbone:
                model_name = self.task+'_model_'+self.datasetname+'_'+model+'_'+bone
                model_names.append(model_name)
                model_path = os.path.join(self.save_model_path, model_name)
                model_paths.append(model_path)
                model_config = {'name':'Segmodel',
                                'model_name':model,
                                'encoder':bone,
                                'activation':'softmax',
                                'init_by_imagenet':False}
                model_configs.append(model_config)
                model_label.append(model+'-'+bone)
        
        for swin_model in mymodels:
            model_name = self.task + '_' + swin_model + '_' + self.datasetname
            model_names.append(model_name)
            model_path = os.path.join(self.save_model_path, model_name)
            model_paths.append(model_path)
            
        model_configs.extend(mymodelconfigs)
        # model label in the figure
        model_label.extend(['Swin-T', 'Swin-S','Swin-M','SDFormer','Swin-L'])
        
        return model_names, model_paths, model_configs, model_label
    

def noise_test(noise_config, best_model=True):
    logger_name= os.path.join(noise_config.noise_result_path, 'noise_test_result.log')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)
    #sh = logging.StreamHandler()
    #sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=logger_name, when='D',backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(th)
    
    segtask=True
    if noise_config.task=='gen':
        segtask=False
    print('starting noise test...')
    noise_test_result = {'noise_level': noise_config.noise_lv}
    for i in range(len(noise_config.model_names)):
        model_name = noise_config.model_names[i]
        model_path = noise_config.model_paths[i]
        model_config = noise_config.model_configs[i]
        model_label = noise_config.model_label[i]
        model_mean_iu = []
        # build model
        net = build_model(config=model_config, 
                          inchannel=noise_config.input_channel, 
                          n_class=noise_config.n_classes, 
                          segtask=segtask)
        print('loading the {} model'.format(model_name))
        if best_model:
            filename = os.path.join(model_path, model_name + 'checkpoint-best.pth')
        else:
            filename = os.path.join(model_path, model_name + 'last-epoch.pth')
        ckpt = torch.load(filename,map_location=torch.device('cpu'))
        epoch_start = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
        net.to(device=noise_config.device)
        net.eval()
        
        logger.info('Noise test result information')
        logger.info('model name: {}'.format(model_name))
        for noise_lv in noise_config.noise_lv:
            result_path = os.path.join(noise_config.noise_result_path, model_name, 'noise_lv_{}%'.format(int(100*noise_lv)))
            img_paths = create_result_dirs(result_path)
            noise_dataset = beam_dataloader(datapath=noise_config.val_path,
                                            class_num=noise_config.n_classes,
                                            noise_level=noise_lv)
            print('testing noise level of {}%'.format(int(100*noise_lv)))
            acc, acc_cls, iu, mean_iu = test_net(net, noise_config, noise_dataset, img_paths)
            model_mean_iu.append(mean_iu)
            logger.info('noise level: {}%'.format(int(100*noise_lv)))
            logger.info('acc = {:.6f}% || meanIOU = {:.6f} || mean acc cls = {:.6f}%'.format(100*acc, mean_iu, 100*acc_cls))
            del noise_dataset
        noise_test_result[model_label] = model_mean_iu
        del net
    print('end noise test')
    return noise_test_result
            
             
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def create_result_dirs(path):
    gt_path = os.path.join(path, 'gt')
    input_path = os.path.join(path, 'input')
    xx_path = os.path.join(input_path, 'xx')     
    yy_path = os.path.join(input_path, 'yy')
    xy_path = os.path.join(input_path, 'xy')
    color_path = os.path.join(input_path, 'color')
    pred_path = os.path.join(path, 'predict')
    
    paths = [gt_path, input_path, xx_path, yy_path, xy_path, color_path, pred_path]
    
    for dirs in paths:
        create_path(dirs)
    
    print('there have been created the dirs in this path')
    
    return paths      

def plot_result(config, result, model_list, result_label):
    save_filename = os.path.join(config.noise_result_path, 'noise_test_result_of_{}.png'.format(result_label))
    noise_lvs = result['noise_level']
    plt.figure(figsize=(20,20))
    font = {'family': 'Times New Roman', 'style': 'normal', 'weight': 'normal', 'size': 28}
    for model in model_list:
        plt.plot(noise_lvs, result[model], label=model, linewidth=2)
    plt.xlabel('Noise Level', fontdict=font)
    plt.ylabel('MIoU', fontdict=font)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.ylim(0,1)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    #plt.xticks(np.arange(1,50,1))
    plt.legend(prop=font)
    plt.savefig(save_filename)
    #plt.grid()
    plt.show()

def main(opt):
    # create model
    noise_config = noise_test_config(opt)
    noise_test_result = noise_test(noise_config=noise_config)
    print(noise_config.model_label[3:19:5])
    model_list = noise_config.model_label[3:19:5]
    plot_result(noise_config, noise_test_result, model_list, 'SDFormer')



if __name__ == "__main__":
    opt = config()
    main(opt)