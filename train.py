"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import os
import time
import copy
import numpy
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

import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss, LovaszLoss

from utils.configs import config
from utils.metric import IOUMetric
from utils.utils import *

from model.SegModel import SegModel, test_model, SegTransformer, SegSwinTransformer, build_model

from dataset.data_loader import beam_dataloader, beam_gendataloader

import cv2

#import warnings


def train_seg_net(net, opt, writer, 
              dataset, optimizer, 
              scheduler, criterion,
              epoch_start=0,
              epochs=20, 
              model_name='seg_model'):
    # create a tensorboard
    train_step = 0
    val_step = 0
    train_losses = []
    val_losses = []
    mious = []
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_iou = 0
    dataset_size = len(dataset)
    #dataset_size = 100
    indices = list(range(dataset_size))
    split = int(np.floor(opt.val_rate * dataset_size))
    if opt.shuffle_dataset :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    scaler = GradScaler() 
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                                            sampler=train_sampler, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                    sampler=valid_sampler, num_workers=8, pin_memory=True)
    if opt.val_path is not None:
        val_dataset = beam_dataloader(opt.val_path, opt.n_classes)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True)
    # logger
    logger_name= os.path.join(opt.train_log, opt.model_name+'train.log')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)
    #sh = logging.StreamHandler()
    #sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=logger_name, when='D',backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)
    #logger.addHandler(sh)
    logger.addHandler(th)

    start_time = time.time()
    # 训练epochs次
    best_epoch = 0
    for epoch in range(epoch_start, epoch_start + epochs):
        print('training '+str(epoch)+'th step...')
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        # train
        idx = 0
        load_time = time.time()
        for image, label in tqdm(train_loader):
            if opt.datasetindex==1:
                image = image[:,:,3:-3,4:-4]
                label = label[:,:,3:-3,4:-4]
            #optimizer.zero_grad()
            train_time = time.time()
            train_loader_size = train_loader.__len__()
            if train_step % opt.show_iter == 0:
                train_input = deformation_to_img(image)
                train_label_color = label2color(label, opt.n_classes, opt.color_array)
            # 将数据拷贝到device中
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            #print(image)
            optimizer.zero_grad()
            if opt.use_fp16:
                with autocast():
                    # 使用网络参数，输出预测结果
                    pred = net(image)
                    if train_step % opt.show_iter == 0:
                        train_pred_color = label2color(pred, opt.n_classes, opt.color_array)
                        writer.add_image(model_name+'/train_input_img/xx', train_input[0:1,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_input_img/yy', train_input[1:2,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_input_img/xy', train_input[2:3,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_ground_truth', train_label_color, global_step=train_step)
                        writer.add_image(model_name+'/train_predict_image', train_pred_color, global_step=train_step)
                    # 计算loss
                    loss = criterion(pred, label)
                writer.add_scalar(model_name+'Loss/train', loss.item(), train_step)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net(image)
                if train_step % opt.show_iter == 0:
                    train_pred_color = label2color(pred, opt.n_classes, opt.color_array)
                    writer.add_image(model_name+'/train_input_img/xx', train_input[0:1,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_input_img/yy', train_input[1:2,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_input_img/xy', train_input[2:3,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_ground_truth', train_label_color, global_step=train_step)
                    writer.add_image(model_name+'/train_predict_image', train_pred_color, global_step=train_step)
                # 计算loss
                loss = criterion(pred, label)
                writer.add_scalar(model_name+'Loss/train', loss.item(), train_step)
                if train_step % 10 == 0:
                    train_losses.append(loss.item())
                # 更新参数
                loss.backward()
                optimizer.step()
            #print(pred)
            scheduler.step(epoch + idx / train_loader_size)
            
            total_time=time.time() - start_time
            mins = total_time//60
            sec = total_time%60
            hours = int(mins//60)
            mins = int(mins%60)
            infomation = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Loss: {:.8f} || train item: {:>5d} / {:>5d} || item time: {:.3f} sec || train time: {:.3f} sec'.format(
                epoch, epoch_start + opt.epochs, hours, mins, sec, loss.item(), train_step%((dataset_size-split)//opt.batch_size), (dataset_size-split)//opt.batch_size, time.time()-load_time, time.time()-train_time)
            if train_step % 100==0:
                logger.info(infomation)
                
            idx = idx+1
            train_step+=1
            load_time = time.time()
            
        # validation
        net.eval()
        iou=IOUMetric(opt.n_classes)
        val_img_index = 0
        pred_used_times = []
        for image, label in tqdm(validation_loader):
            #optimizer.zero_grad()
            val_time = time.time()
            if opt.datasetindex==1:
                image = image[:,:,3:-3,4:-4]
                label = label[:,:,3:-3,4:-4]
            # 将数据拷贝到device中
            #if val_step % opt.show_inter == 0:
            test_input = deformation_to_img(image)
            label_color = label2color(label, opt.n_classes, opt.color_array)
            """
            if n_classes==1:
                n,c,h,w = label.shape
                label = label.reshape(n,h,w)
            """
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            
            # 使用网络参数，输出预测结果
            pred = net(image)
            
            pred_used_time = time.time()-val_time
            pred_used_times.append(pred_used_time)
            
            #if val_step % opt.show_inter == 0:
            pred_color = label2color(pred, opt.n_classes, opt.color_array)
            writer.add_image(model_name+'/train_input_img/xx', test_input[0:1,:,:], global_step=train_step)
            writer.add_image(model_name+'/train_input_img/yy', test_input[1:2,:,:], global_step=train_step)
            writer.add_image(model_name+'/train_input_img/xy', test_input[2:3,:,:], global_step=train_step)
            writer.add_image(model_name+'/ground_truth', label_color, global_step=val_step)
            writer.add_image(model_name+'/predict_image', pred_color, global_step=val_step)
            
            if epoch%opt.save_iter==0:
                # transpose the image channel
                pred_color = pred_color.transpose((1,2,0))
                test_input = test_input.transpose((1,2,0))
                label_color = label_color.transpose((1,2,0))
                h,w,c = pred_color.shape
                pred_color = cv2.resize(pred_color, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
                test_input = cv2.resize(test_input, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
                label_color = cv2.resize(label_color, (int(w*4), int(h*4)),interpolation=cv2.INTER_NEAREST)
                # save seg img
                filename = 'seg_val_{:3d}th_epoch_{:5d}.png'.format(epoch, val_img_index)
                gt_file = os.path.join(opt.ground_truth_path, filename)
                #in_file = os.path.join(opt.input_image_path, filename)
                xx_file = os.path.join(opt.xx_img_path, filename)
                yy_file = os.path.join(opt.yy_img_path, filename)
                xy_file = os.path.join(opt.xy_img_path, filename)
                col_file = os.path.join(opt.color_img_path, filename)
                pred_file = os.path.join(opt.predict_path, filename)
                
                cv2.imwrite(gt_file, label_color)
                #cv2.imwrite(in_file, test_input)
                cv2.imwrite(xx_file, test_input[:,:,0])
                cv2.imwrite(yy_file, test_input[:,:,1])
                cv2.imwrite(xy_file, test_input[:,:,2])
                cv2.imwrite(col_file, test_input)
                cv2.imwrite(pred_file, pred_color)
            
            # 计算loss
            val_loss = criterion(pred, label)
            
            writer.add_scalar(model_name+'Loss/val', val_loss.item(), val_step)
            val_losses.append(val_loss.item())
            val_step = val_step+1 
            
            pred=pred.cpu().data.numpy()
            label = label.cpu().data.numpy()
            if opt.n_classes==1:
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
            val_img_index += 1
            #validation_loss = validation_loss + val_loss.item()
            
        mean_pred_time = np.mean(pred_used_times)
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        mious.append(mean_iu)
        writer.add_scalar(model_name+'meanIOU', mean_iu, epoch)
        logger.info('meanIOU: {:.5f} || acc: {:.5f} || mean acc cls: {:.5f} || mean predict time: {:.8f} sec'.format(mean_iu, acc, acc_cls, mean_pred_time))
        #validation_loss = validation_loss/split
        #print('Validation Loss: ', validation_loss)
        
        # save model
        state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = os.path.join(opt.save_model_path, model_name + 'last-epoch.pth')
        torch.save(state, filename)

        if epoch%opt.save_iter == 0 and epoch>opt.min_iter:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        if mean_iu > best_iou:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mean_iu
            best_epoch = epoch
    logger.info('best MeanIOU: {:.6f} in epoch: {}'.format(best_iou, epoch))
    
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    mious = np.array(mious)
    
    trainlossfile = os.path.join(opt.train_log, 'trainloss.npy')
    vallossfile = os.path.join(opt.train_log, 'valloss.npy')
    miousfile = os.path.join(opt.train_log, 'mious.npy')
    
    np.save(trainlossfile, train_losses)
    np.save(vallossfile, val_losses)
    np.save(miousfile, mious)

def train_gen_net(net, opt, writer,
              dataset, optimizer, 
              scheduler, criterion, 
              epoch_start=0,
              epochs=20,
              model_name='gen_model'):
    # create a tensorboard
    train_step = 0
    val_step = 0
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    
    dataset_size = len(dataset)
    #dataset_size = 100
    indices = list(range(dataset_size))
    split = int(np.floor(opt.val_rate * dataset_size))
    if opt.shuffle_dataset :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    scaler = GradScaler() 
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                                            sampler=train_sampler, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                    sampler=valid_sampler, num_workers=8, pin_memory=True)
    
    # logger
    logger_name= os.path.join(opt.train_log, opt.model_name+'_gen_train.log')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)
    #sh = logging.StreamHandler()
    #sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=logger_name, when='D',backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)
    #logger.addHandler(sh)
    logger.addHandler(th)

    # create folder
    gt_xx = os.path.join(opt.ground_truth_path,'xx')
    gt_yy = os.path.join(opt.ground_truth_path,'yy')
    gt_xy = os.path.join(opt.ground_truth_path,'xy')

    pred_xx = os.path.join(opt.predict_path, 'xx')
    pred_yy = os.path.join(opt.predict_path, 'yy')
    pred_xy = os.path.join(opt.predict_path, 'xy')

    paths = [gt_xx, gt_yy, gt_xy, pred_xx, pred_yy, pred_xy]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    start_time = time.time()
    # 训练epochs次
    for epoch in range(epoch_start, epoch_start + epochs):
        print('training '+str(epoch)+'th step...')
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        # train
        idx = 0
        load_time = time.time()
        for image, label in tqdm(train_loader):
            if opt.datasetindex==1:
                image = image[:,:,3:-3,4:-4]
                label = label[:,:,3:-3,4:-4]
            #optimizer.zero_grad()
            train_time = time.time()
            train_loader_size = train_loader.__len__()
            if train_step % opt.show_iter == 0:
                train_label_color = deformation_to_img(label)
            # 将数据拷贝到device中
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            
            optimizer.zero_grad()
            if opt.use_fp16:
                with autocast():
                    # 使用网络参数，输出预测结果
                    pred = net(image)
                    if train_step % opt.show_iter == 0:
                        train_pred_color = deformation_to_img(pred)
                        writer.add_image(model_name+'/train_ground_truth_img/xx', train_label_color[0:1,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_ground_truth_img/yy', train_label_color[1:2,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_ground_truth_img/xy', train_label_color[2:3,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_predict_image_img/xx', train_pred_color[0:1,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_predict_image_img/yy', train_pred_color[1:2,:,:], global_step=train_step)
                        writer.add_image(model_name+'/train_predict_image_img/xy', train_pred_color[2:3,:,:], global_step=train_step)
                    # 计算loss
                    loss = criterion(pred, label)
                writer.add_scalar(model_name+'Loss/train', loss.item(), train_step)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net(image)
                if train_step % opt.show_iter == 0:
                    train_pred_color = deformation_to_img(pred)
                    writer.add_image(model_name+'/train_ground_truth_img/xx', train_label_color[0:1,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_ground_truth_img/yy', train_label_color[1:2,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_ground_truth_img/xy', train_label_color[2:3,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_predict_image_img/xx', train_pred_color[0:1,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_predict_image_img/yy', train_pred_color[1:2,:,:], global_step=train_step)
                    writer.add_image(model_name+'/train_predict_image_img/xy', train_pred_color[2:3,:,:], global_step=train_step)
                # 计算loss
                loss = criterion(pred, label)
                # 更新参数
                loss.backward()
                optimizer.step()
            scheduler.step(epoch + idx / train_loader_size)
            
            total_time=time.time() - start_time
            mins = total_time//60
            sec = total_time%60
            hours = int(mins//60)
            mins = int(mins%60)
            
            infomation = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Loss: {:.8f} || train item: {:>5d} / {:>5d} || item time: {:.3f} sec || train time: {:.3f} sec'.format(
                epoch, epoch_start + opt.epochs, hours, mins, sec, loss.item(), train_step%((dataset_size-split)//opt.batch_size), (dataset_size-split)//opt.batch_size, time.time()-load_time, time.time()-train_time)
            
            if train_step % 100==0:
                logger.info(infomation)
            idx = idx+1
            train_step+=1
            load_time = time.time()
               
        # validation
        val_img_index = 0
        net.eval()
        pred_used_times = []
        val_losses = []
        for image, label in tqdm(validation_loader):
            #optimizer.zero_grad()
            val_time = time.time()
            if opt.datasetindex==1:
                image = image[:,:,3:-3,4:-4]
                label = label[:,:,3:-3,4:-4]
            # 将数据拷贝到device中
            #if val_step % opt.show_inter == 0:
            label_color = deformation_to_img(label)
            """
            if n_classes==1:
                n,c,h,w = label.shape
                label = label.reshape(n,h,w)
            """
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            
            # 使用网络参数，输出预测结果
            pred = net(image)
            
            pred_used_time = time.time()-val_time
            pred_used_times.append(pred_used_time)
            
            #if val_step % opt.show_inter == 0:
            pred_color = deformation_to_img(pred)
            writer.add_image(model_name+'/ground_truth_img/xx', label_color[0:1,:,:], global_step=val_step)
            writer.add_image(model_name+'/ground_truth_img/yy', label_color[1:2,:,:], global_step=val_step)
            writer.add_image(model_name+'/ground_truth_img/xy', label_color[2:3,:,:], global_step=val_step)
            writer.add_image(model_name+'/predict_image_img/xx', pred_color[0:1,:,:], global_step=val_step)
            writer.add_image(model_name+'/predict_image_img/yy', pred_color[1:2,:,:], global_step=val_step)
            writer.add_image(model_name+'/predict_image_img/xy', pred_color[2:3,:,:], global_step=val_step)
            
            if epoch%opt.save_iter==0:
                # transpose the image channel
                pred_color = pred_color.transpose((1,2,0))
                #test_input = test_input.transpose((1,2,0))
                label_color = label_color.transpose((1,2,0))
                
                # save seg img
                filename = 'gen_val_{:3d}th_epoch_{:5d}.png'.format(epoch, val_img_index)
                gt_xx_file = os.path.join(gt_xx, filename)
                gt_yy_file = os.path.join(gt_yy, filename)
                gt_xy_file = os.path.join(gt_xy, filename)
                #in_file = os.path.join(opt.input_image_path, filename)
                pred_xx_file = os.path.join(pred_xx, filename)
                pred_yy_file = os.path.join(pred_yy, filename)
                pred_xy_file = os.path.join(pred_xy, filename)
                
                cv2.imwrite(gt_xx_file, label_color[:,:,0])
                cv2.imwrite(gt_yy_file, label_color[:,:,1])
                cv2.imwrite(gt_xy_file, label_color[:,:,0])
                #cv2.imwrite(in_file, test_input)
                cv2.imwrite(pred_xx_file, pred_color[:,:,0])
                cv2.imwrite(pred_yy_file, pred_color[:,:,1])
                cv2.imwrite(pred_xy_file, pred_color[:,:,2])
            
            val_loss = criterion(pred, label)
            
            writer.add_scalar(model_name+'Loss/val', val_loss.item(), val_step)
            val_losses.append(val_loss.item())
            val_step = val_step+1
            val_img_index += 1
            
            #validation_loss = validation_loss + val_loss.item()
        
        mean_pred_time = np.mean(pred_used_times)
        mean_val_loss = np.mean(val_losses)
        logger.info('mean loss: {:.5f} || mean predict time: {:.8f} sec'.format(mean_val_loss, mean_pred_time))
        
        # save model
        state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = os.path.join(opt.save_model_path, model_name + 'last-epoch.pth')
        torch.save(state, filename)
        
        if epoch%opt.save_iter == 0 and epoch>opt.min_iter:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        
        if mean_val_loss <= best_loss:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = mean_val_loss
        #validation_loss = validation_loss/split
        #print('Validation Loss: ', validation_loss)

def train_net(net, opt, writer, 
              dataset, optimizer, 
              scheduler, criterion,
              epoch_start=0,
              epochs=20, 
              model_name='seg_model'):
    if opt.is_seg:
        train_seg_net(net=net, opt=opt, dataset=dataset, 
                      writer=writer, optimizer=optimizer,
                      scheduler=scheduler, criterion=criterion, 
                      epoch_start=epoch_start, epochs=epochs,
                      model_name=model_name)
    else:
        train_gen_net(net=net, opt=opt, dataset=dataset, 
                      writer=writer, optimizer=optimizer,
                      scheduler=scheduler, criterion=criterion, 
                      epoch_start=epoch_start, epochs=epochs,
                      model_name=model_name)

def main(opt):
    # create a tensorboard
    writer = SummaryWriter(opt.train_log)

    model_config = opt.configs[opt.model_index]
    segtask=True
    if opt.task=='gen':
        segtask=False
    net = build_model(config=model_config, inchannel=opt.input_channel, n_class=opt.n_classes, segtask=segtask)

    epoch_start = 0
    if opt.is_pretrain:
        print('load pretrain model')
        filename = os.path.join(opt.save_model_path, opt.model_name + 'checkpoint-best.pth')
        ckpt = torch.load(filename)
        epoch_start = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
        #optimizer.load_state_dict(ckpt['optimizer'])
    net.to(device=opt.device)
    
    if segtask:
        dataset = beam_dataloader(opt.train_path, opt.n_classes)
    else:
        dataset = beam_gendataloader(opt.train_path, class_num=opt.n_classes, min_mask_point_num=opt.gen_min_mask, mask_point_num=opt.gen_mask_num)
    #gendataset = beam_gendataloader(datapath=opt.datapath, labelpath=opt.labelpath, min_mask_point=opt.gen_min_mask, mask_point_num=opt.gen_mask_num)
    
    # 定义RMSprop算法
    #optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    #optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay, amsgrad=False)
    optimizer = optim.AdamW(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    #optimizer = optim.Adadelta(net.parameters(), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 定义Loss算法
    if segtask:
        DiceLoss_fn=DiceLoss(mode='multilabel')
        #SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
        Focal_loss = FocalLoss(mode='multilabel')
        #Lovasz_Loss = LovaszLoss(mode='multilabel')
        #criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.5, second_weight=0.5).cuda()
        main_loss = L.JointLoss(first=DiceLoss_fn, second=Focal_loss, first_weight=0.5, second_weight=0.5).cuda()
    else:
        main_loss = nn.MSELoss()
    #criterion = L.JointLoss(first=DiceLoss_fn, second=Lovasz_Loss, first_weight=0.5, second_weight=0.5).cuda()
    
    train_net(net=net, opt=opt, 
              writer=writer, 
              dataset=dataset, 
              optimizer=optimizer, 
              scheduler=scheduler, 
              criterion=main_loss,
              epoch_start=epoch_start, 
              epochs=opt.epochs,
              model_name=opt.model_name)



if __name__ == "__main__":
    #warnings.filterwarnings('ignore')
    opt = config()
    
    main(opt)