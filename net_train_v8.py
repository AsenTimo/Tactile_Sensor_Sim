import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# --- 全局参数 ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 3000

# Top-K 策略参数
TOP_K_EPOCH = 500   # 前 500 Epoch 使用普通 Loss 预热
TOP_K_RATIO = 0.3

# 位置编码参数
POS_ENC_SCALES = [1.0, 10.0, 30.0, 100.0]  # 多尺度频率

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Global: Using device: {device}')

# --- 1. 多尺度位置编码 ---
class MultiScalePositionalEncoding(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scales=None):
        super().__init__()
        if scales is None:
            scales = [1.0, 10.0, 30.0, 100.0]
        
        self.scales = scales
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = len(scales) * mapping_size * 2
        
        # 为每个尺度创建独立的B矩阵
        self.B_matrices = nn.ParameterList()
        for scale in scales:
            B = torch.randn(input_dim, mapping_size) * scale
            # 将scale转换为整数或替换点号为下划线
            scale_name = f'B_{int(scale)}' if scale.is_integer() else f'B_{str(scale).replace(".", "_")}'
            self.register_buffer(scale_name, B)
            self.B_matrices.append(nn.Parameter(B, requires_grad=False))
    
    def forward(self, x):
        batch_size, num_points, _ = x.shape
        encodings = []
        
        for B in self.B_matrices:
            x_proj = 2 * np.pi * x @ B
            enc = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            encodings.append(enc)
        
        encoded = torch.cat(encodings, dim=-1)
        return encoded

# --- 2. 自适应Top-K损失 ---
class AdaptiveTopKLoss(nn.Module):
    def __init__(self, start_ratio=0.8, end_ratio=0.2, transition_epochs=1000):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.transition_epochs = transition_epochs
    
    def get_k_ratio(self, epoch):
        if epoch < self.transition_epochs:
            progress = epoch / self.transition_epochs
            k_ratio = self.start_ratio - (self.start_ratio - self.end_ratio) * progress
        else:
            k_ratio = self.end_ratio
        return max(self.end_ratio, min(self.start_ratio, k_ratio))
    
    def forward(self, pred, target, epoch):
        # 计算每个点的L1误差
        point_errors = torch.sum(torch.abs(pred - target), dim=2)
        
        # 自适应K比例
        k_ratio = self.get_k_ratio(epoch)
        batch_size, num_points = point_errors.shape
        k = max(1, int(num_points * k_ratio))
        
        # Top-K损失
        topk_values, _ = torch.topk(point_errors, k, dim=1)
        loss = topk_values.mean()
        
        return loss, k_ratio

# --- 3. 模型类 (多尺度位置编码 + 注意力) ---
class AdvancedDisplacementPredictor(nn.Module):
    def __init__(self, num_points: int = 80, pos_enc_scales=None):
        super().__init__()
        self.num_points = num_points
        
        # 多尺度位置编码
        self.pos_encoder = MultiScalePositionalEncoding(scales=pos_enc_scales)
        pos_enc_dim = self.pos_encoder.output_dim
        
        # 特征维度
        self.features_dim = 256
        
        # 编码器
        self.encoder_2d = nn.Sequential(
            nn.Linear(pos_enc_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.features_dim),
            nn.BatchNorm1d(self.features_dim),
            nn.ReLU(),
        )
        self.encoder_original = self.encoder_2d
        self.encoder_deformed = self.encoder_2d
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.features_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        self.attention_norm = nn.LayerNorm(self.features_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.features_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, original_markers, deformed_markers):
        batch_size = original_markers.size(0)
        num_points = original_markers.size(1)
        
        # 位置编码
        enc_orig = self.pos_encoder(original_markers)
        enc_def = self.pos_encoder(deformed_markers)
        
        # Flatten
        orig_flat = enc_orig.view(-1, enc_orig.size(-1))
        def_flat = enc_def.view(-1, enc_def.size(-1))
        
        # Encode
        feat_orig = self.encoder_original(orig_flat)
        feat_def = self.encoder_deformed(def_flat)
        
        # Reshape
        feat_orig = feat_orig.view(batch_size, num_points, self.features_dim)
        feat_def = feat_def.view(batch_size, num_points, self.features_dim)
        
        # 注意力
        feat_def_att, _ = self.attention(feat_def, feat_orig, feat_orig)
        feat_def = self.attention_norm(feat_def + feat_def_att)
        
        # Global Pooling
        global_orig = feat_orig.max(dim=1).values
        global_def = feat_def.max(dim=1).values
        
        # Concat Globals
        global_combined = torch.cat((global_orig, global_def), dim=1)
        global_expanded = global_combined.unsqueeze(1).expand(-1, num_points, -1)
        
        # Local + Global Fusion
        final_features = torch.cat((feat_def, global_expanded), dim=2)
        
        # Decode
        final_flat = final_features.reshape(-1, self.features_dim * 3)
        outputs_flat = self.decoder(final_flat)
        
        displacement_3d = outputs_flat.view(batch_size, num_points, 3)
        return displacement_3d

# --- 4. 数据集类 (与v7一致) ---
class MarkerDisplacementDataset(Dataset):
    def __init__(self, marker_2d_path: str, marker_3d_path: str, 
                 original_markers_2d: np.ndarray, original_markers_3d: np.ndarray,
                 stats_2d=None, stats_disp=None):
        df_2d = pd.read_csv(marker_2d_path)
        df_3d = pd.read_csv(marker_3d_path)
        
        if len(df_2d) != len(df_3d):
            raise ValueError(f"行数不一致")
        
        self.markers_2d = df_2d[['u', 'v']].values.astype(np.float32)
        current_3d = df_3d[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
        
        self.original_markers_2d = original_markers_2d.astype(np.float32)
        self.original_markers_3d = original_markers_3d.astype(np.float32)
        
        self.target_displacement = current_3d - self.original_markers_3d
        
        self.stats_2d = stats_2d 
        self.stats_disp = stats_disp
    
    def __len__(self):
        return 1 
    
    def __getitem__(self, idx):
        original_t = torch.tensor(self.original_markers_2d, dtype=torch.float32)
        deformed_input_t = torch.tensor(self.markers_2d, dtype=torch.float32)
        target_t = torch.tensor(self.target_displacement, dtype=torch.float32)
        
        if self.stats_2d is not None:
            mean_2d = torch.tensor(self.stats_2d[0], dtype=torch.float32)
            std_2d = torch.tensor(self.stats_2d[1], dtype=torch.float32)
            original_t = (original_t - mean_2d) / std_2d
            deformed_input_t = (deformed_input_t - mean_2d) / std_2d
            
        if self.stats_disp is not None:
            mean_disp = torch.tensor(self.stats_disp[0], dtype=torch.float32)
            std_disp = torch.tensor(self.stats_disp[1], dtype=torch.float32)
            target_t = (target_t - mean_disp) / std_disp
        
        return original_t, deformed_input_t, target_t

# --- 辅助函数 ---
def compute_normalization_stats(data_list):
    all_data = np.concatenate(data_list, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std

def denormalize_3d(normalized_data, stats):
    if stats is None: return normalized_data
    mean, std = stats[0], stats[1]
    if isinstance(normalized_data, torch.Tensor):
        mean_t = torch.tensor(mean, dtype=normalized_data.dtype, device=normalized_data.device)
        std_t = torch.tensor(std, dtype=normalized_data.dtype, device=normalized_data.device)
        return normalized_data * std_t + mean_t
    else:
        return normalized_data * std + mean

def calculate_accuracy(predicted_disp, target_disp, thresholds):
    if predicted_disp.size == 0: return {}
    distances = np.linalg.norm(predicted_disp - target_disp, axis=2)
    total_points = distances.size
    results = {}
    for t in thresholds:
        correct = np.sum(distances < t)
        results[f'< {t:.2f}mm'] = (correct / total_points) * 100 if total_points > 0 else 0.0
    return results

# --- 主程序 ---
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    marker_2d_dir = 'marker_2D'
    marker_3d_dir = 'marker_3D'
    accuracy_thresholds_global = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    print("===== STARTING TRAINING (ADVANCED POS_ENC + ADAPTIVE TOP-K) =====")
    
    marker_2d_files = sorted(glob.glob(os.path.join(marker_2d_dir, '*.csv')))
    marker_3d_files = sorted(glob.glob(os.path.join(marker_3d_dir, '*.csv')))
    
    if len(marker_2d_files) == 0:
        print("错误: 没有找到数据文件")
        exit()

    # --- 1. 加载参考数据 ---
    original_2d_path = 'markers_points_2d.csv'
    original_3d_path = 'marker_points.csv'
    
    if not os.path.exists(original_2d_path):
        print("Error: Reference files not found.")
        exit()
        
    ref_2d = pd.read_csv(original_2d_path)[['u', 'v']].values.astype(np.float32)
    ref_3d_df = pd.read_csv(original_3d_path)
    if 'x' in ref_3d_df.columns:
        ref_3d = ref_3d_df[['x', 'y', 'z']].values.astype(np.float32)
    else:
        ref_3d = ref_3d_df[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
    num_points = len(ref_2d)

    # --- 2. 预扫描数据计算统计信息 ---
    print("Pre-scanning for stats...")
    all_2d = [ref_2d]
    all_disp = []
    valid_files = []
    
    for f2, f3 in zip(marker_2d_files, marker_3d_files):
        try:
            if os.path.basename(f2) != os.path.basename(f3): continue
            d2 = pd.read_csv(f2)[['u', 'v']].values.astype(np.float32)
            d3 = pd.read_csv(f3)[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
            all_2d.append(d2)
            all_disp.append(d3 - ref_3d)
            valid_files.append((f2, f3))
        except: continue
            
    stats_2d = compute_normalization_stats(all_2d)
    stats_disp = compute_normalization_stats(all_disp)
    print(f"Stats - 2D Mean: {stats_2d[0]}, Disp Std: {stats_disp[1]}")
    
    # --- 3. 创建数据集 ---
    datasets = []
    for f2, f3 in valid_files:
        datasets.append(MarkerDisplacementDataset(f2, f3, ref_2d, ref_3d, stats_2d, stats_disp))
        
    idx = list(range(len(datasets)))
    train_idx, temp_idx = train_test_split(idx, test_size=(1-TRAIN_RATIO), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(TEST_RATIO/(VAL_RATIO+TEST_RATIO)), random_state=42)
    
    train_ds = ConcatDataset([datasets[i] for i in train_idx])
    val_ds = ConcatDataset([datasets[i] for i in val_idx])
    test_ds = ConcatDataset([datasets[i] for i in test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 4. 模型与策略 ---
    model = AdvancedDisplacementPredictor(num_points=num_points, pos_enc_scales=POS_ENC_SCALES).to(device)
    
    # 自适应Top-K损失
    criterion_base = nn.L1Loss()
    criterion_topk = AdaptiveTopKLoss(start_ratio=0.8, end_ratio=TOP_K_RATIO, transition_epochs=TOP_K_EPOCH)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 5. 训练循环 ---
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_accuracy_by_threshold = {f'< {t:.2f}mm': [] for t in accuracy_thresholds_global}
    val_accuracy_by_threshold = {f'< {t:.2f}mm': [] for t in accuracy_thresholds_global}
    
    pbar = tqdm(range(NUM_EPOCHS), desc='Training', ncols=140)
    best_val_mae = float('inf')
    
    for epoch in pbar:
        # Loss Switching Strategy
        use_adaptive_topk = (epoch >= TOP_K_EPOCH)
        
        # Train
        model.train()
        run_loss, cnt = 0.0, 0
        train_preds, train_targets = [], []
        
        for orig, curr, target_disp in train_loader:
            orig, curr, target_disp = orig.to(device), curr.to(device), target_disp.to(device)
            optimizer.zero_grad()
            pred = model(orig, curr)
            
            if use_adaptive_topk:
                loss, k_ratio = criterion_topk(pred, target_disp, epoch)
            else:
                loss = criterion_base(pred, target_disp)
            
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            cnt += 1
            
            with torch.no_grad():
                p_real = denormalize_3d(pred, stats_disp)
                t_real = denormalize_3d(target_disp, stats_disp)
                train_preds.append(p_real.cpu().numpy())
                train_targets.append(t_real.cpu().numpy())
        
        train_losses.append(run_loss/cnt)
        scheduler.step()
        
        # Calc Train Acc
        if train_preds:
            tp_np = np.concatenate(train_preds)
            tt_np = np.concatenate(train_targets)
            train_maes.append(np.mean(np.abs(tp_np - tt_np)))
            t_acc = calculate_accuracy(tp_np, tt_np, accuracy_thresholds_global)
            for k,v in t_acc.items(): 
                train_accuracy_by_threshold[k].append(v)
        
        # Validation
        model.eval()
        val_loss, val_cnt = 0.0, 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for orig, curr, target_disp in val_loader:
                orig, curr, target_disp = orig.to(device), curr.to(device), target_disp.to(device)
                pred = model(orig, curr)
                
                if use_adaptive_topk:
                    loss, k_ratio = criterion_topk(pred, target_disp, epoch)
                else:
                    loss = criterion_base(pred, target_disp)
                
                val_loss += loss.item()
                val_cnt += 1
                
                p_real = denormalize_3d(pred, stats_disp)
                t_real = denormalize_3d(target_disp, stats_disp)
                val_preds.append(p_real.cpu().numpy())
                val_targets.append(t_real.cpu().numpy())
        
        val_losses.append(val_loss/val_cnt)
        
        # Calc Val Acc
        if val_preds:
            vp_np = np.concatenate(val_preds)
            vt_np = np.concatenate(val_targets)
            avg_val_mae = np.mean(np.abs(vp_np - vt_np))
            val_maes.append(avg_val_mae)
            v_acc = calculate_accuracy(vp_np, vt_np, accuracy_thresholds_global)
            for k,v in v_acc.items(): 
                val_accuracy_by_threshold[k].append(v)
            
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), 'best_model_advanced.pth')
        
        loss_name = 'Adaptive-TopK' if use_adaptive_topk else 'L1'
        pbar.set_postfix({
            'Loss': loss_name,
            'TL': f'{train_losses[-1]:.4f}',
            'VL': f'{val_losses[-1]:.4f}',
            'VMae': f'{val_maes[-1]:.3f}',
            'Acc<0.05': f"{val_accuracy_by_threshold['< 0.05mm'][-1]:.1f}%"
        })

    print("\n===== FINAL EVALUATION (TEST SET) =====")
    model.eval()
    test_losses = []
    test_preds, test_targets = [], []
    with torch.no_grad():
        for orig, curr, target_disp in tqdm(test_loader, desc='Test Evaluation', ncols=100):
            orig, curr, target_disp = orig.to(device), curr.to(device), target_disp.to(device)
            pred = model(orig, curr)
            
            loss = criterion_base(pred, target_disp)
            test_losses.append(loss.item())
            
            p_real = denormalize_3d(pred, stats_disp)
            t_real = denormalize_3d(target_disp, stats_disp)
            test_preds.append(p_real.cpu().numpy())
            test_targets.append(t_real.cpu().numpy())
            
    if test_preds:
        tp_np = np.concatenate(test_preds)
        tt_np = np.concatenate(test_targets)
        test_mae = np.mean(np.abs(tp_np - tt_np))
        test_acc = calculate_accuracy(tp_np, tt_np, accuracy_thresholds_global)
        test_acc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in test_acc.items()])
        
        print(f"Test Loss (L1): {np.mean(test_losses):.6f}")
        print(f"Test MAE: {test_mae:.4f}mm")
        print(f"Test Accuracy: [{test_acc_str}]")

    # --- 6. 训练过程可视化 (与v7完全相同) ---
    if NUM_EPOCHS > 0:
        plt.figure(figsize=(18, 10))
        
        # 1. Loss 曲线 (Linear Scale)
        plt.subplot(2, 3, 1)
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train', linewidth=2)
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Val', linewidth=2)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5, label='Top-K Start')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. MAE 曲线
        plt.subplot(2, 3, 2)
        plt.plot(range(1, NUM_EPOCHS + 1), train_maes, label='Train', linewidth=2)
        plt.plot(range(1, NUM_EPOCHS + 1), val_maes, label='Val', linewidth=2)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5, label='Top-K Start')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error (MAE) [mm]')
        plt.title('MAE vs. Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Training Accuracy
        plt.subplot(2, 3, 3)
        for k, v in train_accuracy_by_threshold.items():
            if len(v) > 0:
                val_label = k.split('<')[1].split('mm')[0].strip()
                plt.plot(range(1, NUM_EPOCHS + 1), v, label=f'Train < {val_label}mm', linewidth=2)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Set Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Validation Accuracy
        plt.subplot(2, 3, 4)
        for k, v in val_accuracy_by_threshold.items():
            if len(v) > 0:
                val_label = k.split('<')[1].split('mm')[0].strip()
                plt.plot(range(1, NUM_EPOCHS + 1), v, label=f'Val < {val_label}mm', linewidth=2)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Set Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Loss Contrast (Log Scale)
        plt.subplot(2, 3, 5)
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train', alpha=0.7)
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Val', alpha=0.7)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 6. MAE Contrast
        plt.subplot(2, 3, 6)
        plt.plot(range(1, NUM_EPOCHS + 1), train_maes, label='Train', alpha=0.7)
        plt.plot(range(1, NUM_EPOCHS + 1), val_maes, label='Val', alpha=0.7)
        plt.axvline(x=TOP_K_EPOCH, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('MAE [mm]')
        plt.title('MAE Contrast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        plt.suptitle(f"Training Metrics (Advanced PosEnc + Adaptive Top-K)", fontsize=16, y=1.02)
        plt.show()
    
    # 保存配置
    torch.save({
        'num_points': num_points,
        'model_type': 'Advanced_PosEnc_AdaptiveTopK',
        'top_k_epoch': TOP_K_EPOCH,
        'top_k_ratio': TOP_K_RATIO,
        'pos_enc_scales': POS_ENC_SCALES,
        'stats_2d': stats_2d,
        'stats_disp': stats_disp
    }, 'model_config_advanced.pth')
    
    print("Script execution finished.")