"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import pandas as pd
import numpy as np
import os
import tqdm
from config import config

def save_result(csvfile, save_file, h=64, w=64):
    data = pd.read_csv(csvfile)
    elementlabel = data.columns[4]
    data.sort_values(by=elementlabel, inplace=True)
    e11label, e22label, e12label = data.columns[-3], data.columns[-2], data.columns[-1]
    E11 = data[e11label].values.reshape((h,w))[np.newaxis,:, :]
    E22 = data[e22label].values.reshape((h,w))[np.newaxis,:, :]
    E12 = data[e12label].values.reshape((h,w))[np.newaxis,:, :]
    E = np.concatenate((E11,E22,E12), axis=0)
    #file = csvfile.replace('.csv', '.npy')
    np.save(save_file, E)
    return E

def gen_img_data(csv_path, save_path, h=64, w=64):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in tqdm.tqdm(os.listdir(csv_path)):
        csvfile = os.path.join(csv_path, file)
        npyfile = file.replace('.csv', '.npy')
        npyfile = os.path.join(save_path, npyfile)
        E = save_result(csvfile=csvfile, save_file=npyfile, h=h, w=w)
    
def main(config):
    h, w = config.table_size
    gen_img_data(config.data_path, config.img_data_path, h=h, w=w)


if __name__ == "__main__":
    config = config()
    main(config)
        