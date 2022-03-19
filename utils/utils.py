"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import torch
import numpy as np
import traceback

def label2color(label, n_classes=10, color_array=None):
    if len(label.shape)==4:
        img = get_img(label, n_classes)
    elif len(label.shape)==3:
        img = label
        if torch.is_tensor(img):
            img = img.cpu()
            img = img.numpy()
        img = img-1
        img = img[0,:,:]
        img = img.reshape(img.shape[0], img.shape[1])
        #print(img.shape)
    else:
        raise ValueError('label shape should be 4 (N, C, H, W) or 3(N, H, W)')
    img = get_color(img,c=n_classes, color_array=color_array)
    return img

def get_img(class_possibility, n_classes=10):
    if type(class_possibility) is np.ndarray:
        if n_classes==1:
            img[img>=0.5] = 1
            img[img<0.5] = 0
        else:
            img = np.argmax(class_possibility, axis=1)
    elif torch.is_tensor(class_possibility):
        if n_classes==1:
            img = class_possibility.cpu()
            img = img.data.numpy()
            img[img>=0.5] = 1
            img[img<0.5] = 0
            n,c,h,w =img.shape
            img = img.reshape(n,h,w)
        else:
            img = torch.argmax(class_possibility, dim=1)
            img = img.cpu()
            img = img.data.numpy()
    else:
        print(traceback.format_exc())
        raise TypeError('img should be np.ndarray or torch.tensor')
    img = img[0,:,:]
    img = img.reshape(img.shape[0], img.shape[1])
    return img

def get_color(image, c=10, color_array=None):
    assert c<=10
    if color_array is None:
        color_array = np.random.randint(255, size=(c,3))
    '''
    color_array = np.array([[177, 191, 122],  # farm_land
                            [0, 128, 0],  # forest
                            [128, 168, 93],  # grass
                            [62, 51, 0],  # road
                            [128, 128, 0],  # urban_area
                            [128, 128, 128],  # countryside
                            [192, 128, 0],  # industrial_land
                            [0, 128, 128],  # construction
                            [132, 200, 173],  # water
                            [128, 64, 0]],  # bareland
                           dtype='uint8')
    '''
    color_image = np.zeros((3,image.shape[0], image.shape[1]))
    if c==1:
        for i in range(3):
            color_image[i,:,:] = image*128
    else:
        for idx in range(c):
            add_img = np.tile(color_array[idx], (image.shape[0], image.shape[1], 1)).transpose((2, 0, 1))*(image==idx)
            color_image = color_image + add_img
    return color_image

def deformation_to_img(data):
    if torch.is_tensor(data):
        data = data.cpu().data.numpy()
    if len(data.shape)==4:
        data = data[0]
    c, h, w = data.shape
    flatten_data = data.reshape((-1,c))
    '''
    # normalization
    data_mean = flatten_data.mean(axis=0)
    data_std = flatten_data.std(axis=0)
    flatten_data = (flatten_data-data_mean)/data_std
    flatten_data = flatten_data*0.15+0.5
    flatten_data = np.around(np.clip(flatten_data, 0, 1)*255)
    '''
    data_max = flatten_data.max(axis=0)
    data_min = flatten_data.min(axis=0)
    flatten_data = (flatten_data-data_min)/(data_max-data_min)
    flatten_data = np.round(flatten_data*255)
    #img = flatten_data.reshape((h, w, c))
    img = flatten_data.reshape((c, h, w))
    
    return img


