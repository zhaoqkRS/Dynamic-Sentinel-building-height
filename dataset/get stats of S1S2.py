# %% Functions
# -*- encoding: utf-8 -*-
'''
@Time    :   2025/02/11 20:46:20
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description: 
'''
import os
import sys
import glob
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import warnings
warnings.filterwarnings('ignore')
import json



def get_train_s1s2_stats(s1_dirs, s2_dirs, savepath):
    size = 128
    s1_paths = []
    for dir in s1_dirs:
        paths_dir = glob.glob(dir+'\\*.tif')
        s1_paths += paths_dir
        print(len(paths_dir))


    s2_paths = []
    for dir in s2_dirs:
        paths_dir = glob.glob(dir+'\\*.tif')
        s2_paths += paths_dir
        print(len(paths_dir))


    num_images = len(s2_paths)
    print('S2', len(s2_paths))
    R = np.zeros([num_images, size, size])
    G = np.zeros([num_images, size, size])
    B = np.zeros([num_images, size, size])
    N = np.zeros([num_images, size, size])
    for i, path in tqdm(enumerate(s2_paths)):
        try:
            ds = gdal.Open(path)
            R[i,:,:] = ds.GetRasterBand(1).ReadAsArray()
            G[i,:,:] = ds.GetRasterBand(2).ReadAsArray()
            B[i,:,:] = ds.GetRasterBand(3).ReadAsArray()
            N[i,:,:] = ds.GetRasterBand(4).ReadAsArray()
        except:
            print(path)

    R = np.where(np.isinf(R) | (R < -10000), np.nan, R)
    G = np.where(np.isinf(G) | (G < -10000), np.nan, G)
    B = np.where(np.isinf(B) | (B < -10000), np.nan, B)
    N = np.where(np.isinf(N) | (N < -10000), np.nan, N)

    print(f'===Red=== mean:{np.nanmean(R)}, std:{np.nanstd(R)}')
    print(f'===Green=== mean:{np.nanmean(G)}, std:{np.nanstd(G)}')
    print(f'===Blue=== mean:{np.nanmean(B)}, std:{np.nanstd(B)}')
    print(f'===Nir== mean:{np.nanmean(N)}, std:{np.nanstd(N)}')



    num_images = len(s1_paths)
    print('S1', len(s1_paths))
    VV = np.zeros([num_images, size, size])
    VH = np.zeros([num_images, size, size])
    for i, path in tqdm(enumerate(s1_paths)):
        try:
            ds = gdal.Open(path)
            VV[i,:,:] = ds.GetRasterBand(1).ReadAsArray()
            VH[i,:,:] = ds.GetRasterBand(2).ReadAsArray()
        except:
            print(path)
    VV = np.where(np.isinf(VV) | (VV < -100000), np.nan, VV)
    VH = np.where(np.isinf(VH) | (VH < -100000), np.nan, VH)
    print(f'===VV=== mean:{np.nanmean(VV)}, std:{np.nanstd(VV)}')
    print(f'===VH=== mean:{np.nanmean(VH)}, std:{np.nanstd(VH)}')


    # save
    data = {
        'red': {'mean': np.nanmean(R), 'std': np.nanstd(R)},
        'green': {'mean': np.nanmean(G), 'std': np.nanstd(G)},
        'blue': {'mean': np.nanmean(B), 'std': np.nanstd(B)},
        'nir': {'mean': np.nanmean(N), 'std': np.nanstd(N)},
        'vv': {'mean': np.nanmean(VV), 'std': np.nanstd(VV)},
        'vh': {'mean': np.nanmean(VH), 'std': np.nanstd(VH)},

    }

    with open(savepath, 'w') as f:
        json.dump(data, f)



get_train_s1s2_stats(s1_dirs = [
                                'China_dataset_Amap\\block_s1_2020\\128',
                               ],
                     s2_dirs = [
                                'China_dataset_Amap\\block_s2_2020\\128',
                                ],
                     savepath = 'China_images_training_stats.json')

