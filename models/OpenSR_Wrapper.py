import torch
import torch.nn as nn
import sys
import os
import gc
from omegaconf import OmegaConf

# --- 自动路径修复逻辑 ---
# 自动寻找并添加 opensr_model 到 sys.path，解决导入问题
def find_and_append_module_path(module_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_depth = 10 
    for _ in range(max_depth):
        potential_path = os.path.join(current_dir, module_name)
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            print(f"OpenSR Wrapper: 成功定位 '{module_name}' 于: {potential_path}")
            return True
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: break
        current_dir = parent_dir
    return False

# 尝试导入，如果失败则自动搜索
try:
    from opensr_model.srmodel import SRLatentDiffusion
except ImportError:
    if find_and_append_module_path("opensr_model"):
        try:
            from opensr_model.srmodel import SRLatentDiffusion
        except ImportError:
            # 最后的尝试：搜 models 文件夹
            if find_and_append_module_path("models"):
                 try:
                     from models.SR.opensr_model.srmodel import SRLatentDiffusion
                 except ImportError: pass
    else:
        class SRLatentDiffusion: pass

class OpenSR_Wrapper(nn.Module):
    def __init__(self, weights_path=None, config_path=None, device='cuda', denorm_mean=None, denorm_std=None):
        """
        OpenSR 模型包装器 (性能优化版)
        
        功能：
        1. 加载 VQ-VAE (Autoencoder)，并自动删除沉重的 Diffusion U-Net 以节省显存。
        2. 处理数据反归一化和 [0,1] 缩放。
        3. 利用 Decoder 将 Latent 特征还原为 512x512 的高质量纹理特征。
        """
        super().__init__()
        self.device = device
        self.first_stage_model = None 
        
        # 1. 配置反归一化参数
        if denorm_mean is not None and denorm_std is not None:
            self.register_buffer('mean', torch.tensor(denorm_mean).view(1, 4, 1, 1))
            self.register_buffer('std', torch.tensor(denorm_std).view(1, 4, 1, 1))
            self.do_denorm = True
        else:
            self.do_denorm = False
            print("OpenSR Wrapper Warning: No denormalization parameters provided!")

        # 2. 加载模型并瘦身
        if config_path is None:
             try:
                 import opensr_model
                 module_path = os.path.dirname(opensr_model.__file__)
                 config_path = os.path.join(module_path, "configs", "config_10m.yaml")
             except: pass

        if config_path and os.path.exists(config_path):
            try:
                # 加载完整模型到 CPU (防止瞬间爆显存)
                config = OmegaConf.load(config_path)
                full_model = SRLatentDiffusion(config, device=device)
                
                if weights_path and os.path.exists(weights_path):
                    print(f"OpenSR: 加载权重 -> {weights_path}")
                    checkpoint = torch.load(weights_path, map_location=device)
                    # 兼容不同版本的权重字典 key
                    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
                    full_model.load_state_dict(state_dict, strict=False)
                
                # --- 核心优化：只保留 Autoencoder ---
                self.first_stage_model = full_model.model.first_stage_model
                self.first_stage_model.to(device)
                self.first_stage_model.eval()
                
                # 冻结参数
                for param in self.first_stage_model.parameters():
                    param.requires_grad = False
                
                # 删除大模型，强制回收内存
                del full_model
                torch.cuda.empty_cache()
                gc.collect()
                print("OpenSR: 模型瘦身完成 (仅保留 VQ-VAE).")
                    
            except Exception as e:
                print(f"OpenSR 初始化失败: {e}")
                self.first_stage_model = None
        else:
            print(f"OpenSR Error: Config file not found at {config_path}")

    def forward_feature(self, x):
        """
        输入: (B, 4, 128, 128) Mean-Std 归一化数据
        输出: (B, 4, 512, 512) 重建的高分辨率特征
        """
        if self.first_stage_model is None:
            return torch.zeros(x.shape[0], 4, 512, 512).to(x.device)

        # --- 1. 数据适配 ---
        if self.do_denorm:
            x = x * self.std + self.mean
        
        x = x / 10000.0
        x = torch.clamp(x, 0.0, 1.0)
        
        # OpenSR VQ-VAE 期望输入范围是 [-1, 1]
        x = 2.0 * x - 1.0

        # --- 2. 特征提取 (Autoencoding) ---
        
        # 先插值到 512x512，欺骗 Encoder
        x_up = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            # Encoder: 512 -> Latent(128)
            encoder_posterior = self.first_stage_model.encode(x_up)
            z = encoder_posterior.mode()
            # Decoder: Latent(128) -> Reconstructed(512)
            # 这一步会恢复出 AI 生成的纹理细节
            rec = self.first_stage_model.decode(z)
        # 输出是 [-1, 1] 的 FP16，转回 FP32 以匹配主网络
        return rec.float()