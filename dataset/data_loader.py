"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
#import tensorflow as tf
import numpy as np
import math
import random
import os

import torch
from torch.utils.data import Dataset
import albumentations as A


class beam_dataloader(Dataset):
    def __init__(self, datapath, class_num=4, noise_level=0):
        super(beam_dataloader, self).__init__()
        self.imgpath = os.path.join(datapath, 'img')
        self.maskpath = os.path.join(datapath, 'mask')
        
        self.imgfiles = os.listdir(self.imgpath)
        self.label_num = class_num

        self.transformer = A.OneOf([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5)])
        
        if noise_level>1:
            noise_level=1
        elif noise_level<0:
            noise_level=0
        self.noise_level = noise_level
        
    def data_preprocessing(self, data, label):
        c, h, w= data.shape
        data = data.transpose(1, 2, 0)
        # data [h w c]
        # label [h w]
        
        flatten_data = data.reshape((-1, c))
        data_mean = flatten_data.mean(axis=0)
        data_std = flatten_data.std(axis=0)
        
        data = (data - data_mean)/data_std
        
        #self.label = self.label.transpose(0, 3, 1, 2)
        #self.label = self.label.reshape(n, h, w)
        
        newlabel = np.zeros((h, w, self.label_num))
        for i in range(self.label_num):
            newlabel[:,:,i] = label==i
        
        del flatten_data, label
        label = newlabel
        # label [h w n]
        transfored = self.transformer(image=data, mask=label)
        data = transfored['image']
        label = transfored['mask']

        data = data.transpose(2, 0, 1)
        data = self.add_random_noise(data)
        # data [c h w]
        label = label.transpose(2, 0, 1)
        # label [n h w]
        
        return data, label
            
    def __getitem__(self, index):
        datafile = os.path.join(self.imgpath, self.imgfiles[index])
        maskfile = os.path.join(self.maskpath, self.imgfiles[index])
        
        data = np.load(datafile)
        label = np.load(maskfile)
        
        data, label = self.data_preprocessing(data=data, label=label)
        
        
        return data, label
    
    def __len__(self):
        return len(self.imgfiles)
    
    def add_random_noise(self, img):
        c, h, w = img.shape
        noise_map = 1+self.noise_level*np.clip(np.random.randn(c, h, w),-1,1)

        return noise_map*img
        



class beam_gendataloader(beam_dataloader):
    def __init__(self, datapath, class_num = 4, noise_level=0, min_mask_point_num=20, mask_point_num=20):
        super(beam_gendataloader, self).__init__(datapath=datapath, class_num=class_num, noise_level=noise_level)
        self.min_mask_point_num = min_mask_point_num
        self.mask_points = mask_point_num
            
    def __getitem__(self, index):
        datafile = os.path.join(self.imgpath, self.imgfiles[index])
        maskfile = os.path.join(self.maskpath, self.imgfiles[index])
        
        data = np.load(datafile)
        label = np.load(maskfile)
        
        data, label = self.data_preprocessing(data=data, label=label)
        
        mask = self.random_mask(data)
        gener_data = data*mask
        
        del label
        
        return gener_data, data
    
    def random_mask(self, data):
        c, h, w = data.shape
        num = self.min_mask_point_num + random.randint(0, self.mask_points)
        hindex = np.random.randint(h-1, size=num)
        windex = np.random.randint(w-1, size=num)
        mask = np.zeros((c,h,w))
        mask[:,hindex,windex]=1
        
        return mask            