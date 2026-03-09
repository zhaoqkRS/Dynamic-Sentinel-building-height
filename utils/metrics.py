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
import numpy as np
import torch
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm_fn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

class Evaluator:
    def __init__(self, output, target, type=''):
        self.target = target  # 保持tensor格式
        self.output = output  # 保持tensor格式
        if type=='classification':
            self._compute_confusion_matrix()
        if type=='regression':
            self.target_np = self.target.cpu().detach().numpy().flatten()
            self.output_np = self.output.cpu().detach().numpy().flatten()
            self.valid_mask = (self.target_np != 0)
            self.target_valid = self.target_np[self.valid_mask]
            self.output_valid = self.output_np[self.valid_mask]

            # print(f"数据统计:")
            # total_points = len(self.target_np)
            # valid_points = len(self.target_valid)
            # zero_ratio = (total_points - valid_points) / total_points
            # print(f"  总点数: {total_points}")
            # print(f"  有效非零点数: {valid_points}")
            # print(f"  0值占比: {zero_ratio:.2%}")
            # print(f"  Target有效值范围: [{np.min(self.target_valid):.4f}, {np.max(self.target_valid):.4f}]")
            # print(f"  Output有效值范围: [{np.min(self.output_valid):.4f}, {np.max(self.output_valid):.4f}]")

    def _compute_confusion_matrix(self):
        # pred = (self.output > 0.5).float()
        pred = self.output
        true = self.target
        self.TP = torch.sum((pred == 1) & (true == 1)).cpu().detach().numpy()
        self.FP = torch.sum((pred == 1) & (true == 0)).cpu().detach().numpy()
        self.TN = torch.sum((pred == 0) & (true == 0)).cpu().detach().numpy()
        self.FN = torch.sum((pred == 0) & (true == 1)).cpu().detach().numpy()


    def _pearsonr(self):
        if len(self.target_valid) == 0:
            return 0.0
        return pearsonr(self.target_valid, self.output_valid)[0]


    def _δ(self):
        if len(self.target_valid) == 0:
            return 0.0
        a, b = self.target_np, self.output_np
        valid_mask = (a != 0) & (b != 0)
        c = np.zeros_like(a)
        ratio_ab, ratio_ba = np.divide(a, b, where=valid_mask), np.divide(b, a, where=valid_mask)
        c[valid_mask] = np.maximum(ratio_ab[valid_mask], ratio_ba[valid_mask])
        return np.nansum((c>0)&(c<1.25**3))/np.nansum(c>0)

    def _rmse(self):
        if len(self.target_valid) == 0:
            return 0.0
        # return torch.sqrt(torch.mean((self.target - self.output) ** 2)).cpu().detach().numpy()
        return np.sqrt(mean_squared_error(self.target_np, self.output_np))

    def _nrmse(self):
        # 使用所有有效数据计算均值，或者明确说明归一化方式
        if len(self.target_valid) == 0:
            return 0.0
        scale = np.nanmean(self.target_np)
        return self._rmse() / scale

    def _mse(self):
        if len(self.target_valid) == 0:
            return 0.0
        return mean_squared_error(self.target_np, self.output_np)

    def _mae(self):
        if len(self.target_valid) == 0:
            return 0.0
        return mean_absolute_error(self.target_np, self.output_np)

    def _accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN) if (self.TP + self.TN + self.FP + self.FN)!=0 else 0

    def _precision(self):
        self.precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) != 0 else 0
        return self.precision

    def _recall(self):
        self.recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) != 0 else 0
        return self.recall

    def _f1_score(self):
        self._precision()
        self._recall()
        return (2 * (self.precision * self.recall) / (self.precision + self.recall)) if (self.precision + self.recall) != 0 else 0

    def _iou(self):
        iou_positive = self.TP / (self.TP + self.FP + self.FN) if (self.TP + self.FP + self.FN) > 0 else 0
        iou_negative = self.TN / (self.TN + self.FP + self.FN) if (self.TN + self.FP + self.FN) > 0 else 0
        return iou_positive, iou_negative, (iou_positive + iou_negative) / 2

    def plot_comparison(self, sample_size=200):
        """绘制有效数据的预测值与真实值散点图"""
        if len(self.target_valid) == 0:
            print("没有有效数据可绘制")
            return

        if len(self.target_valid) > sample_size:
            indices = np.random.choice(len(self.target_valid), sample_size, replace=False)
            target_sample = self.target_valid[indices]
            output_sample = self.output_valid[indices]
        else:
            target_sample = self.target_valid
            output_sample = self.output_valid

        plt.figure(figsize=(10, 6))
        plt.scatter(target_sample, output_sample, alpha=0.5)

        # 添加1:1参考线
        min_val = min(target_sample.min(), output_sample.min())
        max_val = max(target_sample.max(), output_sample.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'True vs Predicted (Valid Data Only)\nR2: {self._r2():.4f}, RMSE: {self._rmse():.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


