"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import numpy as np
import cv2
import os

from config import config
import tqdm

def get_multimask(class_num=4, h=256, w=256, n=8, min_area_rate=0.2, max_area_rate=0.9, 
                  spot_num=80, max_R=5, min_R=1, thresh=0.5, win_size=5):
    multimask = np.zeros((h, w), np.uint8)
    for i in range(class_num-1):
        mask = get_mask(h, w, n=n, min_area_rate=min_area_rate, max_area_rate=max_area_rate, 
                  spot_num=spot_num, max_R=max_R, min_R=min_R, thresh=thresh, win_size=win_size)
        multimask[mask==1]=i+1
    
    return multimask

def get_mask(h=256, w=256, n=8, min_area_rate=0.2, max_area_rate=0.9, spot_num=80, max_R=5, min_R=1, thresh=0.5, win_size=5):
    img = np.zeros((h, w), np.uint8)
    area = h*w
    min_area = round(area*min_area_rate)
    max_area = round(area*max_area_rate)
    init_mask = generate_initial_mask(image=img, n=n, min_area=min_area, max_area=max_area)
    mask = sp_noice_mask(img=init_mask, num=spot_num, max_R=max_R, min_R=min_R, thresh=thresh, win_size=win_size)
    return mask

def mask2index(mask):
    h, w = mask.shape
    listmask = mask.reshape(h*w)
    index = np.argwhere(listmask==1)
    index = index.reshape(len(index))+1
    index = index.tolist()
    return index


def generate_initial_mask(image, n, min_area, max_area):
    area = 0
    row, col= image.shape # 行,列
    #print(row)
    while area<min_area or area>max_area:
        point_set = np.zeros((n, 1, 2), dtype=int)
        for j in range(n):
            c = np.random.randint(0, col-1)
            r = np.random.randint(0, row-1)
            point_set[j, 0, 0] = c
            point_set[j, 0, 1] = r
        hull = []
        hull.append(cv2.convexHull(point_set, False))
        drawing_board = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(drawing_board, hull, -1, (1, 1), -1)
        area = cv2.contourArea(hull[0])
    #cv2.namedWindow("drawing_board", 0), cv2.imshow("drawing_board", drawing_board), cv2.waitKey()

    return drawing_board

def sp_noice_mask(img, num=80, max_R=5, min_R=1, thresh=0.5, win_size=5):
    mask = np.ones_like(img)
    h, w = img.shape
    for i in range(num):
        r = np.random.randint(min_R, max_R)
        hindex = np.random.randint(0, h-1)
        windex = np.random.randint(0, w-1)
        cv2.circle(mask, (windex, hindex), r, (0, 0), -1)
    img = img * mask
    img_median = cv2.medianBlur(img, win_size)
    img_median[img_median>=thresh]=1
    img_median[img_median<thresh]=0

    return img_median


        
def main(config):
    h, w = config.table_size
    for i in tqdm.tqdm(range(config.number_of_data)):
        damage_mask = get_multimask(class_num=config.class_num, h=h, w=w, n=config.angle_num, 
                                    min_area_rate=config.min_area_rate, max_area_rate=config.max_area_rate,
                                    spot_num=config.spot_num, max_R=config.Max_R, min_R=config.min_R, 
                                    thresh=config.thresh, win_size=config.win_size)
        
        mask_file_name = '{:0>5d}.npy'.format(i)
        mask_file_path = os.path.join(config.mask_path, mask_file_name)
        
        np.save(mask_file_path, damage_mask)
        
if __name__ == "__main__":
    config = config()
    print('there have '+ str(config.mask_num)+' masks')
    main(config)