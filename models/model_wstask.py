# -*- encoding: utf-8 -*-
'''
@Time    :   2024/12/27 17:01:14
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description: 灵活超分架构，集成双流编码器、CBAM融合与ASPP上下文增强
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from .SR.HRfuse import HRfeature, HRfuse_residual
import opensr_model # pip install opensr_model
# import mlstac
import math
import matplotlib.pyplot as plt

# --- 注意力模块 (CBAM) ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 降维比率，减少参数
        reduced_planes = max(in_planes // ratio, 4)

        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# --- 融合模块 ---

class FusionBlock(nn.Module):
    """
    S1/S2 特征融合模块: Concat -> Conv -> CBAM
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.cbam = CBAM(out_channels)

    def forward(self, s1_feat, s2_feat):
        cat_feat = torch.cat([s1_feat, s2_feat], dim=1)
        fused = self.conv(cat_feat)
        refined = self.cbam(fused)
        return refined + fused  # 残差连接


# --- 上下文增强模块 (ASPP) ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)


# --- 基础网络组件 ---

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.GELU()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class EfficientUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * (scale ** 2), 3, padding=1, groups=in_ch),
            nn.Conv2d(in_ch * (scale ** 2), out_ch * (scale ** 2), 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x): return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, x_channels, skip_channels, hr_channels, out_channels):
        super().__init__()
        self.conv_x = nn.Conv2d(x_channels, out_channels, 1)
        self.conv_skip = nn.Conv2d(skip_channels, out_channels, 1)
        self.conv_hr = nn.Conv2d(hr_channels, out_channels, 1)
        self.res_block = ResidualBlock(out_channels, out_channels)
        self.activate = nn.GELU()

    def forward(self, x, skip, hr_features):
        fused = self.conv_x(x) + self.conv_skip(skip) + self.conv_hr(hr_features)
        return self.res_block(self.activate(fused))


class AdaptiveBinsHeightHead(nn.Module):
    def __init__(self, encoder_channels=64, decoder_channels=8, n_bins=100, min_height=0, max_height=320):
        super().__init__()
        self.min_height = min_height
        self.max_height = max_height
        self.n_bins = n_bins

        # --- 1. 全局宽度预测 (Global Pooling + Softmax) ---
        # 这种写法绝对稳定，不会出现 NaN 或 0
        self.bin_width_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(True),
            # 输出 n_bins 个宽度（不是 n-1），这样总和才是 max_height
            nn.Linear(256, n_bins)
            # 注意：不加 Softplus，我们在 forward 里用 Softmax
        )

        # --- 2. 像素级分类头 ---
        self.height_classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, 3, padding=1),  # 加宽
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1),  # 加深一层
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, n_bins, 1)
        )

    def forward(self, features, decoder_output):
        # features: [B, C, H, W]
        # decoder_output: [B, C, H, W]

        # 1. 计算宽度比例 (Logits -> Softmax)
        bin_width_logits = self.bin_width_predictor(features)

        # 【核心】Softmax 保证所有宽度为正且和为1
        bin_widths_norm = torch.softmax(bin_width_logits, dim=1)  # [B, n_bins]

        # 2. 映射到物理高度
        # [B, n_bins] -> [B, n_bins, 1, 1]
        bin_widths = (bin_widths_norm * (self.max_height - self.min_height)).unsqueeze(-1).unsqueeze(-1)

        # 3. 计算边缘 (Edges)
        # cumsum 自动处理所有边缘，不需要手动赋值最后一个
        bin_right_edges = torch.cumsum(bin_widths, dim=1)

        # 构造完整 edges [B, N+1, 1, 1]
        batch_size = features.shape[0]
        start_edges = torch.full((batch_size, 1, 1, 1), self.min_height, device=features.device)
        bin_edges_global = torch.cat([start_edges, bin_right_edges + self.min_height], dim=1)

        # 广播到全图 [B, N+1, H, W]
        h, w = decoder_output.shape[2], decoder_output.shape[3]
        bin_edges = bin_edges_global.expand(-1, -1, h, w)

        # 4. 计算中心点
        bin_centers = (bin_edges[:, :-1, ...] + bin_edges[:, 1:, ...]) / 2.0

        # 5. 预测分类概率
        bin_logits = self.height_classifier(decoder_output)
        bin_probs = F.softmax(bin_logits, dim=1)

        # 6. 回归高度
        height_map = torch.sum(bin_probs * bin_centers, dim=1, keepdim=True)

        return {
            'pred': height_map,
            'bin_probs': bin_probs,
            'bin_edges': bin_edges
        }

# --- 双流编码器体系 ---

class SingleStreamEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # 32 -> 64 -> 128 -> 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            ResidualBlock(32, 32)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(ResidualBlock(32, 64), ResidualBlock(64, 64))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128))
        self.center = nn.Sequential(
            ResidualBlock(128, 256, dilation=2),
            ResidualBlock(256, 256, dilation=4)
        )

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(self.pool(c1))
        c3 = self.layer3(self.pool(c2))
        center = self.center(c3)
        return [c1, c2, c3, center]


class DualStreamFusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1_enc = SingleStreamEncoder(in_channels=2)
        self.s2_enc = SingleStreamEncoder(in_channels=4)

        # 逐层融合
        self.fuse1 = FusionBlock(32, 32)
        self.fuse2 = FusionBlock(64, 64)
        self.fuse3 = FusionBlock(128, 128)
        self.fuse_center = FusionBlock(256, 256)

    def forward(self, s1, s2):
        s1_feats = self.s1_enc(s1)
        s2_feats = self.s2_enc(s2)

        f1 = self.fuse1(s1_feats[0], s2_feats[0])
        f2 = self.fuse2(s1_feats[1], s2_feats[1])
        f3 = self.fuse3(s1_feats[2], s2_feats[2])
        center = self.fuse_center(s1_feats[3], s2_feats[3])

        return [f1, f2, f3, center]


class Multitask_Decoder_Optimized(nn.Module):
    def __init__(self, n_bins=100, hr_bands=16):
        super().__init__()
        self.hr_bands = hr_bands

        # ASPP 模块
        self.aspp = ASPP(256, 256, rates=[1, 6, 12, 18])

        # FPN
        # self.fpn_center = nn.Sequential(nn.AdaptiveAvgPool2d((32, 32)), nn.Conv2d(hr_bands, hr_bands, 1))
        self.fpn_c2 = nn.Sequential(nn.AdaptiveAvgPool2d((64, 64)), nn.Conv2d(hr_bands, hr_bands, 1))
        self.fpn_c1 = nn.Sequential(nn.AdaptiveAvgPool2d((128, 128)), nn.Conv2d(hr_bands, hr_bands, 1))

        # Footprint Decoder
        self.up2_fp = EfficientUpsample(256, 128, 2)
        self.dec2_fp = DecoderBlock(128, 64, hr_bands, 64)
        self.up1_fp = EfficientUpsample(64, 64, 2)
        self.dec1_fp = DecoderBlock(64, 32, hr_bands, 32)
        self.final_fp = nn.Conv2d(32, 1, 1)

        # Height Decoder
        self.up2_ht = EfficientUpsample(256, 128, 2)
        self.dec2_ht = DecoderBlock(128, 64, hr_bands, 64)
        self.up1_ht = EfficientUpsample(64, 64, 2)
        self.dec1_ht = DecoderBlock(64, 32, hr_bands, 32)
        self.final_ht = AdaptiveBinsHeightHead(encoder_channels=256, decoder_channels=32, n_bins=n_bins)

    def forward(self, encoded_features, compressed_hr_features):
        c1, c2, c3, center = encoded_features

        # 应用 ASPP
        center = self.aspp(center)

        # hr_center = self.fpn_center(compressed_hr_features)
        hr_c2 = self.fpn_c2(compressed_hr_features)
        hr_c1 = self.fpn_c1(compressed_hr_features)

        # Footprint
        d2_fp = self.dec2_fp(self.up2_fp(center), c2, hr_c2)
        d1_fp = self.dec1_fp(self.up1_fp(d2_fp), c1, hr_c1)
        out_fp = self.final_fp(d1_fp)

        # Height
        d2_ht = self.dec2_ht(self.up2_ht(center), c2, hr_c2)
        d1_ht = self.dec1_ht(self.up1_ht(d2_ht), c1, hr_c1)
        out_ht = self.final_ht(center, d1_ht)

        return out_fp, out_ht


class Model(nn.Module):
    def __init__(self, sr='edsr', device='cuda', n_bins=64):
        super(Model, self).__init__()
        self.sr = sr
        self.relu = nn.GELU()
        # --- SR 模型加载 ---
        if self.sr == 'edsr':
            from .SR.EDSR_Net import EDSR
            self.net_hr = EDSR(up_scale=4, n_colors=3, n_feats=64)
            self.net_hr.load_state_dict(torch.load('models/SR/pretrained/EDSR_x4.pth', weights_only=False), strict=False)
            self.net_hr.train()
            for param in self.net_hr.parameters():
                param.requires_grad = True
            sr_feat_channels = 64
            hr_bands = 8

        elif self.sr == 'ldsrs2':
            config = OmegaConf.load('models/SR/pretrained/config_10m.yaml')
            self.net_hr = opensr_model.SRLatentDiffusion(config, device=device)
            self.net_hr.load_pretrained(weights_file='models/SR/pretrained/opensr-ldsrs2_v1_0_0.ckpt')  # load checkpoint
            self.net_hr.eval()
            assert self.net_hr.training == False, "Model has to be in eval mode."
            sr_feat_channels = 4
            hr_bands = 4

        # elif self.sr == 'sen2sr':
        #     mlstac.download(
        #         file="models/SR/SEN2SRLite_RGBN/mlm.json",
        #         output_dir="models/SR/SEN2SRLite_RGBN",
        #     )
        #     self.net_hr = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)
        #     self.net_hr.eval()
        #     sr_feat_channels = 4
        #     hr_bands = 4

        # --- 双流编码器 ---
        self.encoder = DualStreamFusionEncoder()

        # --- 解码器 ---
        self.decoder = Multitask_Decoder_Optimized(n_bins=n_bins, hr_bands=hr_bands)

        # --- 特征压缩 ---
        self.compress_hrfeat = HRfeature(sr_feat_channels, hr_bands, hr_bands)

        self.final_height_fuse = HRfuse_residual(hr_bands, 1, 4, 1, upscale=4)
        self.final_footprint_fuse = HRfuse_residual(hr_bands, 1, 4, 1, upscale=4)

    def train(self, mode=True):
        """
        重写 train 方法。
        当调用 model.train() 时，强制保持 net_hr 在 eval 模式。
        """
        super().train(mode)  # 先把所有层设为 mode (True)

        # 如果有 SR 模型，强制改回 eval
        if hasattr(self, 'net_hr'):
            self.net_hr.eval()

    def forward(self, x, rgb4sr=None):
        s1_data = x[:, 0:2, :, :]
        s2_data = x[:, 2:6, :, :]
        
        if self.sr == 'edsr':
            hr_features = self.net_hr.forward_feature(s2_data[0:3])
            compressed_hr = self.compress_hrfeat(hr_features)

        elif self.sr == 'ldsrs2':
            with torch.no_grad():
                hr_features = self.net_hr.forward((rgb4sr / 10000).to(torch.float32), sampling_steps=50)
                compressed_hr = hr_features

        elif self.sr == 'sen2sr':
            with torch.no_grad():
                hr_features = self.net_hr((rgb4sr / 10000).to(torch.float32))
                compressed_hr = hr_features

        # --- 可视化调试代码 (开始) --- 取第一个样本，前三个通道 (假定是RGB或者特征的前三维)
        # vis_feat = compressed_hr[0, :, :, :].detach().cpu().numpy()
        # # Min-Max 归一化以便显示
        # plt.figure(figsize=(5, 10))
        # plt.subplot(221)
        # plt.imshow(vis_feat.mean(axis=0))
        # plt.title(f'SR features (2.5m)')
        # plt.axis('off')
        # plt.subplot(222)
        # s2 = s2_data[0, :, :, :].detach().cpu().numpy()
        # plt.imshow(s2.mean(axis=0))
        # plt.title(f'Sentinel-2 (10m)')
        # plt.axis('off')
        # plt.savefig(f'hr_features_{self.sr}.png', dpi=600)
        # plt.close()
        # # --- 可视化调试代码 (结束) ---

        # 双流编码
        feats = self.encoder(s1_data, s2_data)

        # 接收字典
        lr_fp, lr_ht_dict = self.decoder(feats, compressed_hr)

        # 从字典取出预测值
        lr_ht = lr_ht_dict['pred']

        # 整合高分辨率输出
        hr_fp = self.final_footprint_fuse(lr_fp, compressed_hr)
        hr_ht = self.relu(self.final_height_fuse(lr_ht, compressed_hr))

        return {
            'footprint_pred': hr_fp,
            'height_pred': hr_ht,
            'sr_s2': compressed_hr
        }

