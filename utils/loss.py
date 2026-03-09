# -*- encoding: utf-8 -*-
'''
@Time    :   2024/12/27 17:01:14
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description: 升级版损失函数，引入跨年一致性约束，移除足迹边缘约束以防止密集区域破碎
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-6, device='cuda'):
        super().__init__()
        self.eps = eps
        # 拉普拉斯算子，用于提取图像中的高频边缘信息
        self.kernel = torch.tensor([
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]
        ], dtype=torch.float32).to(device)

    def edge_detect(self, x):
        # 确保输入有通道维度
        if x.dim() == 3: x = x.unsqueeze(1)
        edges = F.conv2d(x, self.kernel, padding=1)
        return torch.abs(edges)

    def forward(self, pred, target):
        # 目标边缘
        target_edge = self.edge_detect(target.float())
        # 预测边缘
        pred_edge = self.edge_detect(pred)
        # 计算边缘图之间的L1距离
        edge_loss = F.l1_loss(pred_edge, target_edge, reduction='none')
        return edge_loss.mean()

class base_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def _bce_loss(self, pred, target, pos_weight=None, smoothing=0.15):
        loss = nn.BCELoss(reduction='none')
        # 应用标签平滑
        if smoothing > 0:
            with torch.no_grad():
                # 将 0 变为 smoothing/2, 将 1 变为 1 - smoothing/2
                target = target * (1.0 - smoothing) + 0.5 * smoothing
        if pos_weight is not None:
            weight = torch.zeros_like(target)
            weight = torch.fill_(weight, 1 - pos_weight)
            weight[target > 0] = pos_weight
            return torch.mean(weight * loss(pred.float(), target.float()))
        else:
            return loss(pred.float(), target.float()).mean()

    def _dice_loss(self, pred, true, smooth=1e-5):
        pred = pred.contiguous().view(-1)
        true = true.contiguous().view(-1)
        intersection = (pred * true).sum()
        denominator = pred.sum() + true.sum()
        return 1 - (2. * intersection + smooth) / (denominator + smooth)

    def _weighted_consistency_loss(self, pred_main, pred_aux, weight):
        """
        计算加权一致性损失
        pred_main: 主年份预测 (2020)
        pred_aux: 辅助年份预测 (2015/2024)
        weight: 动态变化权重 (0~1)，1表示没变，0表示剧烈变化
        """
        diff = torch.abs(pred_main - pred_aux)
        weight = F.interpolate(
            weight,
            size=pred_main.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        # 只有在权重高的地方（没变的地方）才惩罚差异
        weighted_diff = diff * weight
        return weighted_diff.mean()


class Loss(base_loss):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.alpha = 2.0
        self.edge_loss = EdgeLoss(device=self.device)
        # self.smooth_kernel = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    def _calculate_s1_change_weight(self, img_t1, img_t2):
        """
        基于 Sentinel-1 (VV, VH) 计算不变概率权重
        img_t1, img_t2: [B, C, H, W] 归一化后的输入特征
        假设前两个通道是 VV 和 VH
        """
        # 1. 提取 Sentinel-1 通道 (假设 index 0=VV, 1=VH)
        s1_t1 = img_t1[:, 0:2, :, :]
        s1_t2 = img_t2[:, 0:2, :, :]

        # 2. 计算绝对差异
        # 物理含义：归一化后的后向散射强度差异
        diff_map = torch.abs(s1_t1 - s1_t2)

        # 3. 融合 VV 和 VH 的差异 (取平均)
        # 这里同时也对 Channel 维度求平均，保持 (B, 1, H, W)
        mean_diff = diff_map.mean(dim=1, keepdim=True)
        # 4. 计算权重 (Soft Mask)
        # 差异越大 -> 指数越负 -> 权重趋近 0 (忽略 Loss)
        # 差异越小 -> 指数趋近 0 -> 权重趋近 1 (计算 Loss)
        weight = torch.exp(-self.alpha * mean_diff)
        # alpha 设大一点 (例如 5.0)，让稍微显著一点的差异也能迅速把权重拉低
        # 差异大 -> exp值小 -> weight小 -> 不计算Loss (允许变化)
        return weight.detach()  # 重要：不让梯度回传给输入图像

    def forward(self, target, out_2020, out_2015, out_2024,
                input_2020, input_2015, input_2024,
                epoch_num, begin_unsupervise_epoch):
        '''Initialize targets'''
        height_true = target['height'].to(self.device)
        footprint_true = target['footprint'].to(self.device)
        road_true = target['road'].to(self.device)
        h_2020 = out_2020['height_pred']
        f_2020 = torch.sigmoid(out_2020['footprint_pred'])

        # Height Loss
        h_loss = F.l1_loss(h_2020, height_true, reduction='mean') * 0.5 + F.mse_loss(h_2020, height_true, reduction='mean') * 0.1

        # Footprint Loss: BCE + Dice

        f_loss = self._bce_loss(f_2020, footprint_true, pos_weight=0.75) * 2 + self._dice_loss(f_2020, footprint_true) * 10 + self.edge_loss(f_2020, footprint_true)

        # Road Constraint loss:
        c_loss = (f_2020 * road_true * (1 - footprint_true)).mean() * 10 if epoch_num >= 3 else torch.tensor(0.0)
        # 惩罚在道路上预测出建筑 # 只有那些标签确实没房子的路，才是我们真正要保护的路

        # 无监督学习 -- Consistency Loss (2015/2024 vs 2020) ---
        # 核心逻辑：模型在不同年份的预测应当一致，特别是在我们已知有建筑的区域.
        if epoch_num >= begin_unsupervise_epoch:
            h_2015 = out_2015['height_pred']
            f_2015 = torch.sigmoid(out_2015['footprint_pred'])
            h_2024 = out_2024['height_pred']
            f_2024 = torch.sigmoid(out_2024['footprint_pred'])
            # === 计算动态权重 ===
            # 比较 2015 vs 2020 的输入特征
            w_15_20 = self._calculate_s1_change_weight(input_2015, input_2020)
            # 比较 2024 vs 2020 的输入特征
            w_24_20 = self._calculate_s1_change_weight(input_2024, input_2020)
            cons_h_15 = self._weighted_consistency_loss(h_2020, h_2015, w_15_20)
            cons_f_15 = self._weighted_consistency_loss(f_2020, f_2015, w_15_20)
            cons_h_24 = self._weighted_consistency_loss(h_2020, h_2024, w_24_20)
            cons_f_24 = self._weighted_consistency_loss(f_2020, f_2024, w_24_20)
            cons_loss = (cons_h_24 + cons_f_24) * 1.0 + (cons_h_15 + cons_f_15) * 1.0
        else:
            cons_loss = torch.tensor(0.0)

        # Total Loss
        total_loss = h_loss + f_loss + c_loss + cons_loss

        return total_loss, h_loss, f_loss, c_loss, cons_loss