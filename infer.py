# %%
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/12/27 17:01:14
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description: 
'''
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import rasterio
import math
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
print(torch.__version__)
print('GPU:',torch.cuda.is_available())
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import math
import torch.nn as nn
import json
from models.model_wstask import Model
from utils.dataset import mean_std_normalization, check_dir
from skimage.morphology import erosion, dilation, square, closing, opening, remove_small_objects
from utils.split_and_merge import *


SR = 'ldsrs2'
data_dir = 'F:'
in_channels = 6
out_channels = 2
epoch_num = 10
device =  'cuda'

log_dir = check_dir(f'logs\\{SR}_Model')
model = Model(sr=SR, device=device).to(device)

model_path = os.path.join(log_dir, f'epoch_{epoch_num}.pth')
checkpoint = torch.load(model_path, map_location=device)
# 如果 checkpoint 是一个包含 'state_dict' 和其他信息（如epoch, optimizer）的字典
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    # 如果文件直接就是 state_dict
    state_dict = checkpoint
# 创建新的状态字典，移除 `module.` 前缀
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k # 移除 `module.`
    new_state_dict[name] = v
# 将处理后的状态字典加载到模型中
model.load_state_dict(new_state_dict, strict=True)
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

with open(f'datasets\\China_images_training_stats.json', 'r') as f:
    stats = json.load(f)

all_values = list(range(1196)) + [f"1196-{i}" for i in range(4)] + [f"1197-{i}" for i in range(4)] + [f"1198-{i}" for i in range(4)]
for year in [2015, 2024]:
    savedir = os.makedirs(f'Results\\{year}\\footprinted_height', exist_ok=True)
    for i in tqdm(all_values):
        try:
            s1_image = rasterio.open(data_dir+f'\\S1_{year}\\S1_{year}_{i}.tif')
            s2_image = rasterio.open(data_dir+f'\\S2_{year}\\S2_{year}_{i}.tif')
            s1_image_data = np.stack([s1_image.read(i) for i in range(1, 2+1)], axis=0)
            s2_image_data = np.stack([s2_image.read(i) for i in range(1, 4+1)], axis=0)
            s1_image_data = np.where(np.isnan(s1_image_data) | np.isinf(s1_image_data), 0, s1_image_data)
            s2_image_data = np.where(np.isnan(s2_image_data) | np.isinf(s2_image_data), 0, s2_image_data)
        except Exception as e:  
            print(e)
            print(i)  
            continue

        # if s2_image_data.mean()>10:
        #     s2_image_data = s2_image_data/10000 训练集是1000多的
        s1_vv_coef_data = mean_std_normalization(np.expand_dims(s1_image_data[0, :, :], axis=0), mean=stats['vv']['mean'], std=stats['vv']['std'])
        s1_vh_coef_data = mean_std_normalization(np.expand_dims(s1_image_data[1, :, :], axis=0), mean=stats['vh']['mean'], std=stats['vh']['std'])
        s2_red_data = mean_std_normalization(np.expand_dims(s2_image_data[0, :, :], axis=0), mean=stats['red']['mean'],std=stats['red']['std'])
        s2_green_data = mean_std_normalization(np.expand_dims(s2_image_data[1, :, :], axis=0), mean=stats['green']['mean'],std=stats['green']['std'])
        s2_blue_data = mean_std_normalization(np.expand_dims(s2_image_data[2, :, :], axis=0), mean=stats['blue']['mean'],std=stats['blue']['std'])
        s2_nir_data = mean_std_normalization(np.expand_dims(s2_image_data[3, :, :], axis=0), mean=stats['nir']['mean'], std=stats['nir']['std'])
        features = np.concatenate([s1_vv_coef_data, s1_vh_coef_data, s2_red_data, s2_green_data, s2_blue_data, s2_nir_data], axis=0)
        features = np.where(np.isnan(features), 0, features)
        # print(features.shape)
        feature_tile_list = split_to_tiles(features, window_size=128, overlap=16)
        
        rgb_features = np.concatenate([np.expand_dims(s2_image_data[0,:,:], axis=0), 
                                       np.expand_dims(s2_image_data[1,:,:], axis=0),
                                        np.expand_dims(s2_image_data[2,:,:], axis=0),
                                        np.expand_dims(s2_image_data[3,:,:], axis=0)], axis=0)
        rgb_features = np.where(np.isnan(rgb_features), 0, rgb_features)
        # print(rgb_features.shape)
        rgb_feature_tile_list = split_to_tiles(rgb_features, window_size=128, overlap=16)

        del s1_vv_coef_data, s1_vh_coef_data, s2_red_data, s2_green_data, s2_blue_data, s2_nir_data, s2_image_data   

        r, c = features.shape[1], features.shape[2]
        del features, rgb_features


        pred_height_tile_list = []
        pred_footprint_tile_list = []
        for feature, rgb_feature in tqdm(zip(feature_tile_list, rgb_feature_tile_list), desc=f"Processing tiles for {i}",
                                         leave=False,  # 内层完成后不保留进度条
                                         position=1,  # 固定位置显示
                                         ):
            image = torch.from_numpy(np.asarray(feature[0])).type(torch.FloatTensor).view(1,6,128,128).to(device)
            rgb = torch.from_numpy(np.asarray(rgb_feature[0])).type(torch.FloatTensor).view(1,4,128,128).to(device)
            output_tile = model(image, rgb)
            pred_height_tile = output_tile['height_pred']
            pred_height_tile = torch.squeeze(pred_height_tile, dim=0).detach().cpu().numpy()
            pred_height_tile_list.append([pred_height_tile, feature[1]])
            pred_footprint_tile = torch.sigmoid(output_tile['footprint_pred'])
            pred_footprint_tile = torch.squeeze(pred_footprint_tile, dim=0).detach().cpu().numpy()
            pred_footprint_tile_list.append([pred_footprint_tile, feature[1]])
        pred_height = merge_tiles(pred_height_tile_list, (1, r*4, c*4), window_size=512, scale_factor=4, crop=10) # overlap*4 = 64, crop是打算保留的像素，64-crop就是要丢弃的
        pred_footprint = merge_tiles(pred_footprint_tile_list, (1, r*4, c*4), window_size=512, scale_factor=4, crop=10) # overlap*4 = 64, crop是打算保留的像素，64-crop就是要丢弃的
        pred_footprint = np.where(pred_footprint>=0.5, 1, 0)
        # del height_class_output_logits_tile, pred_height_tile_list, feature_tile_list

        example = rasterio.open(data_dir+f'\\S1_{year}\\S1_{year}_{i}.tif')
        original_transform = example.transform
        scale_factor = 4
        new_transform = rasterio.Affine(
        original_transform.a / scale_factor,  # x方向分辨率调整
        original_transform.b,                # 旋转参数保持不变
        original_transform.c,                # 左上角x坐标不变
        original_transform.d,                # 旋转参数保持不变
        original_transform.e / scale_factor, # y方向分辨率调整
        original_transform.f                 # 左上角y坐标不变
        )


        meta_ht = {'driver': 'GTiff', 'dtype': 'float32', 'width':4*c, 'height':4*r,
                'transform':new_transform,'count': 1, 'crs': example.crs, 'compress': 'lzw'}

        # meta = {'driver': 'GTiff', 'dtype': 'int32', 'width':4*c, 'height':4*r,
        #         'transform':new_transform,'count': 1, 'crs': example.crs, 'compress': 'lzw'}
        #
        # final_footprint = np.where(pred_height<=0, 0, pred_footprint)
        # final_footprint2 = opening(final_footprint[0], square(2))
        # final_footprint2 = erosion(final_footprint[0], square(3))
        # final_footprint2 = dilation(final_footprint2, square(3))
        # final_footprint2 = dilation(final_footprint2, square(3))
        # final_footprint2 = closing(final_footprint2, square(3))
        final_footprint = opening(pred_footprint[0], square(3))
        final_footprint_rev = final_footprint <=0
        final_footprint_rev = remove_small_objects(final_footprint_rev, min_size=40)
        final_footprint[final_footprint_rev<=0] = 1
        post_footprint = np.expand_dims(final_footprint, axis=0)
        footprinted_height = np.where(post_footprint==1, pred_height, 0)

        # with rasterio.open(savedir+f'\\footprint\\{year}_footrint_{i}.tif','w', **meta) as dst:
        #     dst.write(pred_footprint)

        # with rasterio.open(savedir+f'\\height\\{year}_height_{i}.tif','w', **meta_ht) as dst:
        #     dst.write(pred_height)

        # with rasterio.open(savedir+f'\\postprocessed_footprint\\{year}_postprocessed_footprint_{i}.tif','w', **meta) as dst:
        #     dst.write(post_footprint)

        with rasterio.open(savedir+f'\\{year}_footprinted_height_{i}.tif','w', **meta_ht) as dst:
            dst.write(footprinted_height)


# # %% 查缺
# all_values = list(range(1196)) + [f"1196-{i}" for i in range(4)] + [f"1197-{i}" for i in range(4)] + [f"1198-{i}" for i in range(4)]
# for year in [2015,2024]:
#     for i in tqdm(all_values):
#         fp = f'Results\\{year}\\footprint\\{year}_footprint_{i}.tif'
#         ht = f'Results\\{year}\\height\\{year}_height_{i}.tif'
#         if os.path.exists(fp):
#             continue
#         else:
#             print(os.path.basename(fp))
#         if os.path.exists(ht):
#             continue
#         else:
#             print(os.path.basename(ht))
#
# # %% 计算体积
# import numpy as np
# from tqdm import tqdm
# import rasterio
# from utils.Geotools import savetif
# # all_values = list(range(1196)) + [f"1196-{i}" for i in range(4)] + [f"1197-{i}" for i in range(4)] + [f"1198-{i}" for i in range(4)]
# all_values = [9,15,24,47,78,83,84,155,597,641,713,831,847,935,956,973,989,995,1050,1055,1161,1177]
# for year in ['2015','2024']:
#     for i in tqdm(all_values):
#         vl_savepath = f'Results\\{year}\\volume\\{year}_volume_{i}.tif'
#         # if not os.path.exists(vl_savepath):
#         footprint = rasterio.open(f'Results\\{year}\\footprint\\{year}_footprint_{i}.tif').read(1)
#         height = rasterio.open(f'Results\\{year}\\height\\{year}_height_{i}.tif').read(1)  # 建筑物高度栅格
#         vl = footprint*height/100*2.5*2.5 # unit: m3
#         vl = np.where(footprint!=0, vl, 0)
#         savetif(vl, example=f'Results\\{year}\\footprint\\{year}_footprint_{i}.tif',savepath=vl_savepath)
#
