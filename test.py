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


def test_net(net, opt, dataset):
    if opt.is_pretrain:
        print('load pretrain model')
        filename = os.path.join(opt.save_model_path, opt.model_name + 'checkpoint-best.pth')
        ckpt = torch.load(filename)
        epoch_start = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
        #optimizer.load_state_dict(ckpt['optimizer'])
        #net.to(device=opt.device)
    # 选择设备，有cuda用cuda，没有就用cpu
    net.to(device=opt.device)
    device = opt.device
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    n_classes=opt.n_classes
    net.eval()
    test_step = 0
    iou=IOUMetric(n_classes)
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
        
        file = '{:.5d}.jpg'.format(test_step)
        
        gfile = os.path.join(opt.ground_truth_path, file)
        infile = os.path.join(opt.input_image_path, file)
        predfile = os.path.join(opt.predict_path, file)
        
        label_color=label_color.transpose((1,2,0))
        test_input = test_input.transpose((1,2,0))
        pred_color = pred_color.transpose((1,2,0))
        
        cv2.imwrite(filename=gfile, img=label_color)
        cv2.imwrite(filename=infile, img=test_input)
        cv2.imwrite(filename=predfile, img=pred_color)
        
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
        

def main(opt):
    # create model
    model_config = opt.configs[opt.model_index]
    segtask=True
    if opt.task=='gen':
        segtask=False
    net = build_model(config=model_config, inchannel=opt.input_channel, n_class=opt.n_classes, segtask=segtask)

    epoch_start = 0
    if segtask:
        dataset = beam_dataloader(data_path=opt.val_path, 
                                  class_num=opt.n_classes, 
                                  noise_level=opt.noise_level)
    else:
        dataset = beam_gendataloader(data_path=opt.val_path, 
                                     class_num=opt.n_classes, 
                                     noise_level=opt.noise_level,
                                     min_mask_point_num=opt.gen_min_mask, 
                                     mask_point_num=opt.gen_mask_num)
    
    test_net(net=net, opt=opt, dataset=dataset)



if __name__ == "__main__":
    opt = config()
    main(opt)