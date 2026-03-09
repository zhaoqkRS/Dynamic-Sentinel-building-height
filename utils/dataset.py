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
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage import zoom
import torch.nn as nn
# from osgeo import gdal, gdalconst
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from torchvision import transforms
import torchinfo
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import json
from rasterio import merge
from .histogram_matching import histogram_matching  # 导入直方图匹配模块
import random
# def get_backscatterCoef(raw_s1_dta):
#     coef = np.power(10.0, raw_s1_dta / 10.0)
#     coef = np.where(coef > 1.0, 1.0, coef)
#     return coef
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def mosaic_tif(file_dir, output_path):
    if isinstance(file_dir, list):
        file_paths = file_dir
    else:
        file_paths = glob(file_dir + "/*.tif")# 获取文件夹下所有栅格的路径
    datasets = []
    for path in file_paths:
        src = rasterio.open(path)
        datasets.append(src)
    mosaic_data, mosaic_transform = merge.merge(datasets)
    mosaic_crs = src.crs
    profile = {
        'driver': 'GTiff',
        'height': mosaic_data.shape[1],
        'width': mosaic_data.shape[2],
        'count': 1,
        'dtype': mosaic_data.dtype,
        'crs': mosaic_crs,
        'transform': mosaic_transform,
        'compress': 'lzw'}
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mosaic_data)
        
def remove_abnornal(data):
    return np.where(np.isnan(data) | np.isinf(data) | (data<-10000), np.nan, data)

def mean_std_normalization(arr, mean, std):
    arr = np.array(arr)
    # 注意：这里假设传入的arr已经是处理过异常值或经过直方图匹配的数据
    # 如果是直方图匹配后的数据，可能不需要再次remove_abnormal，除非匹配过程引入了异常
    arr = remove_abnornal(arr) 
    normalized_arr = (arr - mean) / std
    return normalized_arr


class RSDataset(Dataset):
    def __init__(self, dataset='China', stat_path='', num_sample=5000, oversample=True):
        if dataset == 'China':
            dataset_folder = 'China_dataset_Amap'
        else:
            print('No such dataset')
        s1_dir_2015 = f'datasets\\{dataset_folder}\\block_s1_2015\\128'
        s2_dir_2015 = f'datasets\\{dataset_folder}\\block_s2_2015\\128'
        s1_dir_2020 = f'datasets\\{dataset_folder}\\block_s1_2020\\128'
        s2_dir_2020 = f'datasets\\{dataset_folder}\\block_s2_2020\\128'
        s1_dir_2024 = f'datasets\\{dataset_folder}\\block_s1_2024\\128'
        s2_dir_2024 = f'datasets\\{dataset_folder}\\block_s2_2024\\128'
        height_dir = f'datasets\\{dataset_folder}\\block_ref_height\\512_closing'
        road_dir = f'datasets\\{dataset_folder}\\block_ref_road\\512'

        # 获取所有文件并排序（确保对应关系）
        def get_sorted_files(directory):
            files = [file for file in glob(os.path.join(directory, '*.tif'))]
            files.sort(key=lambda x: os.path.basename(x))
            return files

        # 获取各路径文件列表
        s1_paths_2015 = get_sorted_files(s1_dir_2015)
        s2_paths_2015 = get_sorted_files(s2_dir_2015)
        s1_paths_2020 = get_sorted_files(s1_dir_2020)
        s2_paths_2020 = get_sorted_files(s2_dir_2020)
        s1_paths_2024 = get_sorted_files(s1_dir_2024)
        s2_paths_2024 = get_sorted_files(s2_dir_2024)
        height_paths = get_sorted_files(height_dir)
        road_paths = get_sorted_files(road_dir)

        # 🎯 核心改进：生成随机索引并应用到所有文件列表
        import random
        random.seed(42)
        all_indices = list(range(len(s1_paths_2015)))
        random.shuffle(all_indices)
        selected_indices = all_indices[:num_sample]
        # 使用相同的随机索引选择文件
        def select_files(file_list, indices):
            return [file_list[i] for i in indices]
        s1_paths_2015 = select_files(s1_paths_2015, selected_indices)
        s2_paths_2015 = select_files(s2_paths_2015, selected_indices)
        s1_paths_2020 = select_files(s1_paths_2020, selected_indices)
        s2_paths_2020 = select_files(s2_paths_2020, selected_indices)
        s1_paths_2024 = select_files(s1_paths_2024, selected_indices)
        s2_paths_2024 = select_files(s2_paths_2024, selected_indices)
        height_paths = select_files(height_paths, selected_indices)
        road_paths = select_files(road_paths, selected_indices)

        print(len(s1_paths_2015), len(s2_paths_2015), len(s1_paths_2020), len(s2_paths_2020), len(s1_paths_2024), len(s2_paths_2024), len(height_paths), len(road_paths))

        assert len(s1_paths_2015) == len(s2_paths_2015) == len(s1_paths_2020) == len(s2_paths_2020) == len(s1_paths_2024) == len(s2_paths_2024) == len(height_paths) == len(road_paths), print('!')

        self.s1_images_data_2015 = self.read_s1_images(s1_paths_2015)
        self.s2_images_data_2015 = self.read_s2_images(s2_paths_2015)
        self.s1_images_data_2020 = self.read_s1_images(s1_paths_2020)
        self.s2_images_data_2020 = self.read_s2_images(s2_paths_2020)
        self.s1_images_data_2024 = self.read_s1_images(s1_paths_2024)
        self.s2_images_data_2024 = self.read_s2_images(s2_paths_2024)
        self.height_data = self.read_singleband_labels(height_paths)
        self.road_data = self.read_singleband_labels(road_paths)

        self.stat_path = stat_path

        self.oversample = oversample
        self.sample_weights = None

        # 计算每个样本的建筑物高度统计信息
        if self.oversample:
            self._compute_building_statistics()
            self._compute_sample_weights()

    def _compute_building_statistics(self):
        """计算每个样本的建筑物高度统计信息"""
        self.building_stats = []

        for idx in trange(len(self.height_data)):
            height_data = self.height_data[idx]

            # 创建有效掩码
            valid_mask = ~(np.isnan(height_data) | np.isinf(height_data) |
                           (height_data < 0) | (height_data > 1000))

            # 提取建筑物像素（高度>0的有效像素）
            building_pixels = height_data[valid_mask & (height_data > 0)]

            if len(building_pixels) > 0:
                # 计算建筑物高度的统计量
                mean_height = np.mean(building_pixels)
                max_height = np.max(building_pixels)
                building_area = len(building_pixels)  # 建筑物像素数量

                # 确定高度类别
                if mean_height >= 45:
                    height_class = 'very-high'
                elif mean_height >= 32:
                    height_class = 'high'
                elif mean_height >= 21:
                    height_class = 'medium-high'
                elif mean_height >= 12:
                    height_class = 'medium'
                elif mean_height >= 4:
                    height_class = 'low'
                else:
                    height_class = 'very-low' # 低层建筑阈值 4层以下
            else:
                mean_height = 0
                max_height = 0
                building_area = 0
                height_class = 'none'

            self.building_stats.append({
                'mean_height': mean_height,
                'max_height': max_height,
                'building_area': building_area,
                'height_class': height_class
            })

    def _compute_sample_weights(self):
        """根据建筑物高度计算样本权重"""
        # 统计每个类别的样本数量
        class_counts = {
            'none': 0, 'very-low':0, 'low': 0, 'medium': 0, 'medium-high':0, 'high': 0, 'very-high': 0
        }

        for stat in self.building_stats:
            class_counts[stat['height_class']] += 1

        print("=== 过采样前类别分布 ===")
        total_samples = len(self.building_stats)
        for cls, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        print(f"总样本数: {total_samples}")

        # 计算逆频率权重
        total_samples = len(self.building_stats)
        self.sample_weights = []

        for stat in self.building_stats:
            cls = stat['height_class']
            if cls == 'very-high':
                weight = total_samples / np.sqrt(class_counts['very-high'])
            elif cls == 'high':
                weight = total_samples / np.sqrt(class_counts['high'])
            elif cls == 'medium-high':
                weight = total_samples / np.sqrt(class_counts['medium-high'])
            elif cls == 'medium':
                weight = total_samples / np.sqrt(class_counts['medium'])
            elif cls == 'low':
                weight = total_samples / np.sqrt(class_counts['low'])
            elif cls == 'very-low':
                weight = total_samples / np.sqrt(class_counts['very-low'])
            else:
                weight = 0.0001  # 无建筑物样本权重最低

            # 考虑建筑物面积的影响（可选）
            # if cls != 'none':
            #     area_factor = min(stat['building_area'] / 1000, 2.0)
            #     weight *= (1 + area_factor)
            # 考虑连片建筑：建筑密度高的样本也给予更高权重
            density_factor = min(stat['building_area'] / 100, 3.0)  # 建筑密度因子
            weight *= (1 + density_factor * 0.3)

            self.sample_weights.append(weight)

        # 归一化权重
        self.sample_weights = np.array(self.sample_weights)
        self.sample_weights = self.sample_weights / np.sum(self.sample_weights)
        # print("权重分布示例:")
        # for i, (weight, stat) in enumerate(zip(self.sample_weights[:10], self.building_stats[:10])):
        #     print(f"  样本{i}: {stat['height_class']}, 权重: {weight:.4f}")

        # 打印权重分布
        print("\n=== 过采样权重设置 ===")
        weight_by_class = {'none': [], 'very-low': [], 'low': [], 'medium': [], 'medium-high':[], 'high': [], 'very-high': []}
        for i, stat in enumerate(self.building_stats):
            weight_by_class[stat['height_class']].append(self.sample_weights[i])

        for cls, weights in weight_by_class.items():
            if weights:
                avg_weight = np.mean(weights)
                print(f"  {cls}平均权重(*1000): {avg_weight*1000:.4f}")

    def read_s1_images(self, s1_paths):
        images = []
        for image_path in s1_paths:
            try:
                rsdl_data = rasterio.open(image_path)
                images.append(np.stack([rsdl_data.read(i) for i in range(1, 2 + 1)], axis=0))
                rsdl_data = None
            except:
                print(image_path)
        return images

    def read_s2_images(self, s2_paths):
        images = []
        for image_path in s2_paths:
            try:
                rsdl_data = rasterio.open(image_path)
                images.append(np.stack([rsdl_data.read(i) for i in range(1, 4 + 1)], axis=0))
                rsdl_data = None
            except:
                print(image_path)
        return images
    
    def read_singleband_labels(self, labels_paths):
        labels = []
        for label_path in labels_paths:
            try:
                rsdl_data = rasterio.open(label_path)
                labels.append(rsdl_data.read(1))
            except:
                print(label_path)
        return labels

    def get_normalized_features(self, s1_data, s2_data, stat_path):
        """
        对输入的S1和S2数据进行归一化并拼接，不再依赖idx从列表读取
        s1_data: [2, H, W] (VV, VH)
        s2_data: [4, H, W] (R, G, B, NIR)
        """
        with open(stat_path, 'r') as f:
            stats = json.load(f)
        
        s1_vv_coef_data = mean_std_normalization(np.expand_dims(s1_data[0,:,:], axis=0), mean=stats['vv']['mean'], std=stats['vv']['std'])
        s1_vh_coef_data = mean_std_normalization(np.expand_dims(s1_data[1,:,:], axis=0), mean=stats['vh']['mean'], std=stats['vh']['std'])
        s2_red_data = mean_std_normalization(np.expand_dims(s2_data[0,:,:], axis=0), mean=stats['red']['mean'], std=stats['red']['std'])
        s2_green_data = mean_std_normalization(np.expand_dims(s2_data[1,:,:], axis=0), mean=stats['green']['mean'], std=stats['green']['std'])
        s2_blue_data = mean_std_normalization(np.expand_dims(s2_data[2,:,:], axis=0), mean=stats['blue']['mean'], std=stats['blue']['std'])
        s2_nir_data = mean_std_normalization(np.expand_dims(s2_data[3,:,:], axis=0), mean=stats['nir']['mean'], std=stats['nir']['std'])
        features = np.concatenate([s1_vv_coef_data, s1_vh_coef_data, s2_red_data, s2_green_data, s2_blue_data, s2_nir_data], axis=0)
        features = np.where(np.isnan(features), 0, features)
        features = torch.from_numpy(features).type(torch.FloatTensor)
        return features

    def _apply_augmentation(self, samples_dict):
        """
        同步对样本字典中的所有张量进行数据增强
        samples_dict 包含: image_2015, image_2020, image_2024, footprint, height, road 等
        """
        # 1. 随机水平翻转
        if random.random() > 0.5:
            for key in samples_dict:
                if isinstance(samples_dict[key], torch.Tensor):
                    samples_dict[key] = torch.flip(samples_dict[key], dims=[2])  #

        # 2. 随机垂直翻转
        if random.random() > 0.5:
            for key in samples_dict:
                if isinstance(samples_dict[key], torch.Tensor):
                    samples_dict[key] = torch.flip(samples_dict[key], dims=[1])  #

        # 3. 随机 90/180/270 度旋转
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            for key in samples_dict:
                if isinstance(samples_dict[key], torch.Tensor):
                    samples_dict[key] = torch.rot90(samples_dict[key], k=rot_k, dims=[1, 2])  #

        return samples_dict

    def __len__(self):
        return len(self.s1_images_data_2015)

    def __getitem__(self, idx):
        ##### building
        # height
        height_data = self.height_data[idx]
        valid_mask = ~(np.isnan(height_data) | np.isinf(height_data) | (height_data < 0) | (height_data > 1000))
        height_data = np.where(valid_mask, height_data, 0)
        del valid_mask

        # footprint
        # 原始分割标签
        footprint_data = np.where(height_data>1, 1, 0).astype(np.uint8)

        # --- 核心修改：形态学闭运算解决密集区域空洞 ---
        # 使用3x3的核（在2.5m分辨率下覆盖约7.5m的区域），足以填补大部分胡同和握手楼的缝隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        footprint_data = cv2.morphologyEx(footprint_data, cv2.MORPH_CLOSE, kernel)
        # ---------------------------------------

        footprint_data_tensor = torch.Tensor(np.expand_dims(footprint_data, axis=0)).type(torch.FloatTensor)
        height_data_tensor = torch.Tensor(np.expand_dims(height_data, axis=0)).type(torch.FloatTensor)

        ###### road
        road_data = self.road_data[idx]
        road_data = np.where((np.isnan(road_data)) | (np.isinf(road_data) | ((road_data!=0)&(road_data!=1))), 0, road_data)
        road_data_tensor = torch.Tensor(np.expand_dims(road_data, axis=0)).type(torch.FloatTensor)

        ###### images
        # 1. 获取原始数据
        raw_s1_2020 = self.s1_images_data_2020[idx]
        raw_s2_2020 = self.s2_images_data_2020[idx]

        raw_s1_2015 = self.s1_images_data_2015[idx]
        raw_s2_2015 = self.s2_images_data_2015[idx]

        raw_s1_2024 = self.s1_images_data_2024[idx]
        raw_s2_2024 = self.s2_images_data_2024[idx]

        # 3. 归一化特征 (使用2020年的统计参数，因为数据已经匹配到2020的分布)
        features_2015 = self.get_normalized_features(raw_s1_2015, raw_s2_2015, self.stat_path)
        features_2020 = self.get_normalized_features(raw_s1_2020, raw_s2_2020, self.stat_path)
        features_2024 = self.get_normalized_features(raw_s1_2024, raw_s2_2024, self.stat_path)

        features_2015 = torch.nan_to_num(features_2015, 0.0)
        features_2020 = torch.nan_to_num(features_2020, 0.0)
        features_2024 = torch.nan_to_num(features_2024, 0.0)

        plot = False
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 12))
            plt.subplot(331)
            # 显示匹配后的2015
            plt.imshow(raw_s1_2015[0,:,:], cmap='gray')
            plt.title('Sentinel-1 2015 (Matched)', fontsize='large')
            plt.subplot(332)
            plt.imshow(raw_s1_2020[0,:,:], cmap='gray')
            plt.title('Sentinel-1 2020', fontsize='large')
            plt.subplot(333)
            # 显示匹配后的2024
            plt.imshow(raw_s1_2024[0,:,:], cmap='gray')
            plt.title('Sentinel-1 2024 (Matched)', fontsize='large')
            plt.subplot(334)
            plt.imshow(raw_s2_2015[0,:,:], cmap='gray')
            plt.title('Sentinel-2 2015', fontsize='large')
            plt.subplot(335)
            plt.imshow(raw_s2_2020[0,:,:], cmap='gray')
            plt.title('Sentinel-2 2020',  fontsize='large')
            plt.subplot(336)
            plt.imshow(raw_s2_2024[0,:,:], cmap='gray')
            plt.title('Sentinel-2 2024', fontsize='large')
            plt.subplot(337)
            plt.imshow(footprint_data_tensor[0,:,:].cpu().detach())
            plt.title('Building footprint', fontsize='large')
            plt.subplot(338)
            plt.imshow(height_data_tensor[0,:,:].cpu().detach())
            plt.title('Building height', fontsize='large')
            plt.subplot(339)
            plt.imshow(road_data_tensor[0,:,:].cpu().detach())
            plt.title('Road', fontsize='large')
            plt.savefig('training_data_vis.png')
            plt.show()


        sample = {"image_2015": features_2015,
                  "image_2020": features_2020,
                  "image_2024": features_2024,
                  "rgb4sr_2015": raw_s2_2015,
                  "rgb4sr_2020": raw_s2_2020,
                  "rgb4sr_2024": raw_s2_2024,
                  "footprint": footprint_data_tensor ,
                  "height": height_data_tensor,
                  "road": road_data_tensor,
                  }
        # sample = self._apply_augmentation(sample)  #
        return sample