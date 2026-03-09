# %% Functions
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/12/27 17:01:14
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description:   
'''
# startup.py
import os
import sys
import random
# 添加Anaconda路径
# conda_path = r"C:\Users\Administrator\Anaconda3\envs\ai"
# os.environ['PATH'] = conda_path + ';' + conda_path + r'\Library\bin;' + os.environ['PATH']
# 现在导入rasterio
import rasterio
from glob import glob
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Add path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from models.model_wstask import Model

from utils.dataset import RSDataset, check_dir
from utils.loss import Loss
from utils.metrics import Evaluator


def train(SR, device, batch_size, in_lr, num_epochs, train_perc, num_sample, begin_unsupervise_epoch=10):
    #########################
    ### Data preparation ####
    #########################
    # Dataset initialization
    dataset = RSDataset(dataset = 'China',
                        stat_path = 'datasets\\China_images_training_stats.json',
                        num_sample = num_sample,
                        oversample = True)

    train_size = int(len(dataset) * train_perc)
    test_size = int(len(dataset) - train_size)
    print(f"Train size: {train_size}, Test size: {test_size}")

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    # Sampler setup
    if dataset.sample_weights is not None:
        train_indices = train_dataset.indices
        train_weights = [dataset.sample_weights[i] for i in train_indices]
        train_weights = np.array(train_weights)
        # 防止除以0或nan
        train_weights = np.nan_to_num(train_weights, nan=0.0)
        train_weights = train_weights / (np.sum(train_weights) + 1e-8)

        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights).type(torch.DoubleTensor),
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=train_shuffle)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #########################
    ### Train model ########
    #########################
    log_dir = check_dir(f'logs\\{SR}_Model')
    model = Model(sr=SR, device=device).to(device)# <--- 关键修改：传入统计参数

    criterion = Loss(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=in_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    log_df = pd.DataFrame()

    # print(f"Start Training with SR module: {SR}...")

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        start = time.time()
        total_train_loss = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs}', mininterval=1.0)

        for i, sample in enumerate(pbar):
            optimizer.zero_grad()
            image_2020 = sample["image_2020"].to(device)
            rgb4sr_2020 = sample['rgb4sr_2020'].to(device)if SR=='ldsrs2' else None
            out_2020 = model(image_2020, rgb4sr_2020)
            if epoch >= begin_unsupervise_epoch:
                image_2015, image_2024 = sample["image_2015"].to(device), sample["image_2024"].to(device)
                rgb4sr_2015 = sample['rgb4sr_2015'].to(device)if SR=='ldsrs2' else None
                rgb4sr_2024 = sample['rgb4sr_2024'].to(device)if SR=='ldsrs2' else None
                out_2015, out_2024 = model(image_2015, rgb4sr_2015), model(image_2024, rgb4sr_2024)
            else:
                image_2015, image_2024 = None, None
                out_2015, out_2024 = None, None
            total_loss, h_loss, f_loss, c_loss, cons_loss = criterion(
                target=sample,
                out_2020=out_2020,
                out_2015=out_2015,
                out_2024=out_2024,
                epoch_num=epoch,
                input_2020=image_2020,
                input_2015=image_2015,
                input_2024=image_2024,
                begin_unsupervise_epoch=begin_unsupervise_epoch
            )

            total_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item(),
                              'ht': h_loss.item(),
                              'fp': f_loss.item(),
                              'rd': c_loss.item(),
                              'tc': cons_loss.item(),
                              })

        scheduler.step()
        avg_train_loss = total_train_loss / len(trainloader)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Time: {(time.time() - start) / 60:.1f} min")
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch+1}.pth'))

        ##################
        ###### validate ######
        ##################
        if (epoch) % 1 != 0:
            continue

        model.eval()
        total_val_loss = 0
        # Metrics
        RMSE, MAE, R, OA, F1, MIOU = 0, 0, 0, 0, 0, 0
        R_C, MAE_C, OA_C, MIOU_C = 0, 0, 0, 0
        actual_val_batches = 0
        max_val_batches = 100
        with torch.no_grad():
            for i, sample in enumerate(tqdm(testloader, desc='Validating')):
                if i > max_val_batches: break
                vis_ref_ht = sample["height"][0, 0, :, :].cpu().detach().numpy()
                vis_ref_fp = sample["footprint"][0, 0, :, :].cpu().detach().numpy()
                vis_ref_rd = sample["road"][0, 0, :, :].cpu().detach().numpy()
                vis_s1_15 = sample['image_2015'][0, 0, :, :].cpu().detach().numpy()
                vis_s1_20 = sample['image_2020'][0, 0, :, :].cpu().detach().numpy()
                vis_s1_24 = sample['image_2024'][0, 0, :, :].cpu().detach().numpy()

                image_2020 = sample["image_2020"].to(device)
                rgb4sr_2020 = sample['rgb4sr_2020'].to(device) if SR == 'ldsrs2' else None
                out_2020 = model(image_2020, rgb4sr_2020)
                image_2015, image_2024 = sample["image_2015"].to(device), sample["image_2024"].to(device)
                rgb4sr_2015 = sample['rgb4sr_2015'].to(device) if SR=='ldsrs2' else None
                rgb4sr_2024 = sample['rgb4sr_2024'].to(device) if SR=='ldsrs2' else None
                out_2015, out_2024 = model(image_2015, rgb4sr_2015), model(image_2024, rgb4sr_2024)
                del rgb4sr_2015, rgb4sr_2020, rgb4sr_2024
 
                # Loss
                val_loss, _, _, _, _ = criterion(sample, out_2020, out_2015, out_2024, image_2020, image_2015, image_2024, epoch, begin_unsupervise_epoch)
                total_val_loss += val_loss.item()

                # 监督学习 Metrics
                h_pred = out_2020['height_pred']
                f_pred = torch.where(torch.sigmoid(out_2020['footprint_pred']) >= 0.5, 1, 0)
                h_true = sample['height'].to(device)
                f_true = sample['footprint'].to(device)

                h_eval = Evaluator(h_pred, h_true, type='regression')
                RMSE += h_eval._rmse()
                MAE += h_eval._mae()
                R += h_eval._pearsonr()

                f_eval = Evaluator(f_pred, f_true, type='classification')
                OA += f_eval._accuracy()
                F1 += f_eval._f1_score()
                _, _, miou = f_eval._iou()
                MIOU += miou

                # 无监督学习 跨年份一致性 Consistency Evaluation
                h_15 = out_2015['height_pred']
                h_24 = out_2024['height_pred']
                height_evaluator_1 = Evaluator(h_15, h_pred, type='regression')
                height_evaluator_2 = Evaluator(h_pred, h_24, type='regression')
                height_evaluator_3 = Evaluator(h_15, h_24, type='regression')
                MAE_C += (height_evaluator_1._mae() + height_evaluator_2._mae() + height_evaluator_3._mae()) / 3
                R_C += (height_evaluator_1._pearsonr() + height_evaluator_2._pearsonr() + height_evaluator_3._pearsonr()) / 3
                f_15 = torch.where(torch.sigmoid(out_2015['footprint_pred']) >= 0.5, 1, 0)
                f_24 = torch.where(torch.sigmoid(out_2024['footprint_pred']) >= 0.5, 1, 0)
                footprint_evaluator_1 = Evaluator(f_15, f_pred, type='classification')
                footprint_evaluator_2 = Evaluator(f_pred, f_24, type='classification')
                footprint_evaluator_3 = Evaluator(f_15, f_24, type='classification')
                OA_C += (footprint_evaluator_1._accuracy() + footprint_evaluator_2._accuracy() + footprint_evaluator_3._accuracy()) / 3
                MIOU_C += (footprint_evaluator_1._iou()[2] + footprint_evaluator_2._iou()[2] + footprint_evaluator_3._iou()[2]) / 3
                # del out_2015, out_2020, out_2024, h_15, h_24, h_pred, f_15, f_24, f_pred
                actual_val_batches += 1

            # Normalize Metrics
            n_batches = actual_val_batches
            val_res = {
                'RMSE': RMSE/n_batches, 'MAE': MAE/n_batches, 'R': R/n_batches,
                'OA': OA/n_batches, 'F1': F1/n_batches, 'MIOU': MIOU/n_batches,
            }
            val_c_res = {
                'MAE_C': MAE_C / n_batches, 'R_C': R_C / n_batches,
                'OA_C': OA_C / n_batches, 'MIOU_C': MIOU_C / n_batches,
            }

            print(f"\nValidation Results Epoch {epoch+1}:")
            print(f"Height -> RMSE: {val_res['RMSE']:.3f}, MAE: {val_res['MAE']:.3f},  R: {val_res['R']:.3f}")
            print(f"Footprint -> OA: {val_res['OA']:.3f}, F1: {val_res['F1']:.3f}, mIOU: {val_res['MIOU']:.3f}")

            if epoch >= begin_unsupervise_epoch:
                print('\n跨年份一致性')
                print(f"Height -> R: {val_c_res['R_C']:.3f}")
                print(f"Height -> MAE: {val_c_res['MAE_C']:.3f}")
                print(f"Footprint -> OA: {val_c_res['OA_C']:.3f}")
                print(f"Footprint -> MIOU: {val_c_res['MIOU_C']:.3f}")


            # Logging
            epoch_log = {'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': total_val_loss/n_batches}
            epoch_log.update(val_res)
            epoch_log.update(val_c_res)
            new_row = pd.DataFrame([epoch_log])
            log_df = pd.concat([log_df, new_row], ignore_index=True)
            log_df.to_csv(os.path.join(log_dir, 'log.csv'), index=False)


            # Visualization
            plt.figure(figsize=(10, 6))
            # image_2015, image_2020, image_2024 = sample["image_2015"].to(device), sample["image_2020"].to(device), sample["image_2024"].to(device)
            # rgb4sr_2015, rgb4sr_2020, rgb4sr_2024 = sample['rgb4sr_2015'].to(device), sample['rgb4sr_2020'].to(device), sample['rgb4sr_2024'].to(device)
            # out_2015, out_2020, out_2024 = model(image_2015, rgb4sr_2015), model(image_2020, rgb4sr_2020), model(image_2024, rgb4sr_2024)

            h_15, h_pred, h_24 = out_2015['height_pred'][0, 0], out_2020['height_pred'][0, 0], out_2024['height_pred'][0, 0]
            sr_s2_2015, sr_s2_2020, sr_s2_2024 = out_2015['sr_s2'], out_2020['sr_s2'], out_2024['sr_s2']
            del out_2015, out_2020, out_2024
            plt.subplot(4, 3, 1)
            plt.imshow(h_15.detach().cpu().numpy(), cmap='RdYlBu_r', vmin=0, vmax=40)
            plt.colorbar()
            plt.axis('off')
            plt.title('Predicted height 2015', fontsize=8)

            plt.subplot(4, 3, 2)
            plt.imshow(h_pred.detach().cpu().numpy(), cmap='RdYlBu_r', vmin=0, vmax=40)
            plt.title('Predicted height 2020', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 3)
            plt.imshow(h_24.detach().cpu().numpy(), cmap='RdYlBu_r', vmin=0, vmax=40)
            plt.title('Predicted height 2024', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 4)
            plt.imshow(sr_s2_2015[0, :, :, :].cpu().detach().numpy().mean(axis=0))
            plt.title('SR Sentinel-2 2015', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 5)
            plt.imshow(sr_s2_2020[0, :, :, :].cpu().detach().numpy().mean(axis=0))
            plt.title('SR Sentinel-2 2020', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 6)
            plt.imshow(sr_s2_2024[0, :, :, :].cpu().detach().numpy().mean(axis=0))
            plt.title('SR Sentinel-2 2024', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 7)
            plt.imshow(vis_s1_15, cmap='gray')
            plt.title('Sentinel-1 2015', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 8)
            plt.imshow(vis_s1_20, cmap='gray')
            plt.title('Sentinel-1 2020', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 9)
            plt.imshow(vis_s1_24, cmap='gray')
            plt.title('Sentinel-1 2024', fontsize=8)
            plt.axis('off')
            plt.colorbar()
            
            plt.subplot(4, 3, 10)
            plt.imshow(vis_ref_ht, cmap='RdYlBu_r', vmin=0, vmax=60)
            plt.title('Building height', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 11)
            plt.imshow(vis_ref_fp, cmap='gray')
            plt.title('Building footprint', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 12)
            plt.imshow(vis_ref_rd, cmap='gray')
            plt.title('Road', fontsize=8)
            plt.axis('off')
            plt.colorbar()

            plt.savefig(os.path.join(log_dir, f'val_vis_epoch_{epoch+1}.png'), dpi=600)
            plt.close()

            del sample

    print("Training Finished.")
    # --- Plotting Training Curves ---
    print("Plotting training curves...")

    # Load log data
    log_path = os.path.join(log_dir, 'log.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        # 1. Loss Curve
        plt.figure(figsize=(8, 7))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        # Val loss might not be present for every epoch if validation frequency > 1
        # Drop NaNs to plot connected lines for validation
        val_df = df.dropna(subset=['val_loss'])
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['val_loss'], label='Val Loss', marker='s')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
        plt.close()

        # 2. Height Metrics (MAE, RMSE)
        plt.figure(figsize=(10, 6))
        val_df = df.dropna(subset=['MAE', 'RMSE'])  # Ensure we plot valid validation steps
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['MAE'], label='MAE (2020)', marker='o')
            plt.plot(val_df['epoch'], val_df['RMSE'], label='RMSE (2020)', marker='s')

            plt.title('Height Estimation Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Meters')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'height_metrics.png'))
            plt.close()

        # 3. Footprint Metrics (F1, mIOU)
        plt.figure(figsize=(10, 6))
        val_df = df.dropna(subset=['F1', 'MIOU'])
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['F1'], label='F1 Score', marker='o')
            plt.plot(val_df['epoch'], val_df['MIOU'], label='mIOU', marker='s')
            plt.title('Footprint Segmentation Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'footprint_metrics.png'))
            plt.close()

        # 4. Consistency Metrics (MAE_Cons, R_Cons) - New!
        plt.figure(figsize=(10, 6))
        # Check if consistency metrics exist in the dataframe
        if 'MAE_C' in df.columns or 'R_C' in df.columns:
            val_df = df.dropna(subset=['MAE_C'])  # Filter valid rows
            if not val_df.empty:
                ax1 = plt.gca()
                ax2 = ax1.twinx()  # Create a second y-axis

                # Plot MAE Consistency on left axis
                line1 = ax1.plot(val_df['epoch'], val_df['MAE_C'], label='Consistency MAE (Lower is Better)',
                                 color='blue', marker='^', linestyle='--')
                ax1.set_ylabel('Consistency MAE (Meters)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')

                # Plot R Consistency on right axis (if available)
                if 'R_C' in val_df.columns:
                    line2 = ax2.plot(val_df['epoch'], val_df['R_C'], label='Consistency R (Higher is Better)',
                                     color='orange', marker='*', linestyle='-.')
                    ax2.set_ylabel('Consistency Correlation (R)', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')

                    # Combine legends
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='upper center')
                else:
                    ax1.legend(loc='upper right')

                plt.title('Temporal Consistency Metrics (2015 vs 2024)')
                ax1.set_xlabel('Epoch')
                ax1.grid(True)
                plt.savefig(os.path.join(log_dir, 'consistency_metrics.png'))
                plt.close()
        else:
            print("Consistency metrics columns not found in log.csv, skipping consistency plot.")

    else:
        print(f"Log file not found at {log_path}, skipping plotting.")