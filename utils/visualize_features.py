import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_features(features, batch_idx=0, save_path=None):
    ####### features # (B, C, H, W)
    features = features[batch_idx].detach().cpu().numpy()  # (C, H, W)
    num_channels = features.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 3))
    if num_channels == 1:
        axes = [axes]  # 确保单通道时axes为列表
    for c in range(num_channels):
        channel_data = features[c]
        # 归一化到 [0,1]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        if max_val != min_val:
            channel_norm = (channel_data - min_val) / (max_val - min_val)
        else:
            channel_norm = channel_data * 0  # 全零通道显示为黑色
        # 显示单通道特征图
        ax = axes[c]
        ax.imshow(channel_norm, cmap='RdYlBu_r')  # 使用颜色映射增强对比度
        ax.set_title(f'Channel {c}')
        ax.axis('off')
    plt.suptitle(f'Compressed HR Features (Batch {batch_idx})', y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()