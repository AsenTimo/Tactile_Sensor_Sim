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
BATCH_SIZE = 64  # Transformer显存占用稍大，稍微调小Batch Size
LEARNING_RATE = 0.0005 
NUM_EPOCHS = 3000

# Top-K 策略参数
TOP_K_EPOCH = 500
TOP_K_RATIO = 0.3

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Global: Using device: {device}')

# --- 1. 损失函数 (Top-K L1) ---
class TopKL1Loss(nn.Module):
    def __init__(self, top_k_ratio=1.0):
        super().__init__()
        self.top_k_ratio = top_k_ratio

    def forward(self, pred, target):
        point_wise_loss = torch.sum(torch.abs(pred - target), dim=2)
        loss_flat = point_wise_loss.view(-1)
        k = int(loss_flat.shape[0] * self.top_k_ratio)
        if k < 1: k = 1
        top_k_loss, _ = torch.topk(loss_flat, k)
        return top_k_loss.mean()

# --- 2. 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: (B, N, 2)
        x_proj = 2 * np.pi * x @ self.B
        # [sin, cos] -> (B, N, 128)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- 3. 模型类 (Transformer Architecture) ---
class MarkerDisplacementPredictor(nn.Module):
    def __init__(self, num_points: int = 80):
        super().__init__()
        self.num_points = num_points
        
        # Transformer 超参数
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1
        
        # 位置编码: 映射到 128 维 (64 sin + 64 cos)
        self.pos_enc_dim = 128 
        self.pos_encoder = PositionalEncoding(input_dim=2, mapping_size=64, scale=10.0)
        
        # 输入投影层
        # 输入是 Original(128) + Deformed(128) = 256 维
        self.input_proj = nn.Linear(self.pos_enc_dim * 2, self.d_model)
        self.ln_in = nn.LayerNorm(self.d_model)
        
        # Transformer Encoder
        # batch_first=True 使得输入格式为 (Batch, Seq_Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 解码
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3) # 输出 dx, dy, dz
        )

    def forward(self, original_markers, deformed_markers):
        # original_markers: (B, N, 2)
        # deformed_markers: (B, N, 2)
        
        # 1. 坐标位置编码
        enc_orig = self.pos_encoder(original_markers) # (B, N, 128)
        enc_def = self.pos_encoder(deformed_markers)  # (B, N, 128)
        
        # 2. 特征融合与投影
        # 将原始状态和变形状态拼接，作为 Transformer 的 token 特征
        combined_feat = torch.cat([enc_orig, enc_def], dim=-1) # (B, N, 256)
        x = self.input_proj(combined_feat) # (B, N, d_model)
        x = self.ln_in(x)
        
        # 3. Transformer处理
        # 这里的Sequence Length就是点的数量N
        x = self.transformer_encoder(x) # (B, N, d_model)
        
        # 4. 输出预测
        output = self.decoder(x) # (B, N, 3)
        
        return output

# --- 4. 数据集类 ---
class MarkerDisplacementDataset(Dataset):
    def __init__(self, marker_2d_path: str, marker_3d_path: str, 
                 original_markers_2d: np.ndarray, original_markers_3d: np.ndarray,
                 stats_2d=None, stats_disp=None):
        
        df_2d = pd.read_csv(marker_2d_path)
        df_3d = pd.read_csv(marker_3d_path)
        
        if len(df_2d) != len(df_3d): raise ValueError(f"行数不一致")
        
        self.markers_2d = df_2d[['u', 'v']].values.astype(np.float32)
        if 'deformed_x' in df_3d.columns:
            current_3d = df_3d[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
        else:
            current_3d = df_3d[['x', 'y', 'z']].values.astype(np.float32)
        
        self.original_markers_2d = original_markers_2d.astype(np.float32)
        self.original_markers_3d = original_markers_3d.astype(np.float32)
        
        self.target_displacement = current_3d - self.original_markers_3d
        
        self.stats_2d = stats_2d 
        self.stats_disp = stats_disp
    
    def __len__(self): return 1 
    
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
    if not data_list: return np.array([0.0,0.0]), np.array([1.0,1.0])
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
    marker_2d_dir = 'marker_2D'
    marker_3d_dir = 'marker_3D'
    accuracy_thresholds_global = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    print("===== STARTING TRAINING (TRANSFORMER + TOP-K L1) =====")
    
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

    # --- 2. 匹配文件并划分数据集索引 ---
    print("Matching files...")
    valid_files = []
    for f2, f3 in zip(marker_2d_files, marker_3d_files):
        if os.path.basename(f2) == os.path.basename(f3): 
            valid_files.append((f2, f3))
    
    idx = list(range(len(valid_files)))
    train_idx, temp_idx = train_test_split(idx, test_size=(1-TRAIN_RATIO), random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(TEST_RATIO/(VAL_RATIO+TEST_RATIO)), random_state=SEED)
    
    print(f"Dataset Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # --- 3. 仅基于训练集计算统计信息 ---
    print("Calculating stats on TRAIN set only...")
    train_2d_data = [ref_2d]
    train_disp_data = []
    
    for i in train_idx:
        f2, f3 = valid_files[i]
        try:
            d2 = pd.read_csv(f2)[['u', 'v']].values.astype(np.float32)
            d3_df = pd.read_csv(f3)
            if 'x' in d3_df.columns:
                d3 = d3_df[['x', 'y', 'z']].values.astype(np.float32)
            else:
                d3 = d3_df[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
            train_2d_data.append(d2)
            train_disp_data.append(d3 - ref_3d)
        except Exception as e:
            print(f"Warning reading {f2}: {e}")

    stats_2d = compute_normalization_stats(train_2d_data)
    stats_disp = compute_normalization_stats(train_disp_data)
    print(f"Stats - 2D Mean: {stats_2d[0]}, Disp Std: {stats_disp[1]}")

    # --- 4. 创建数据集实例 ---
    def create_datasets_from_indices(indices):
        ds_list = []
        for i in indices:
            f2, f3 = valid_files[i]
            ds_list.append(MarkerDisplacementDataset(f2, f3, ref_2d, ref_3d, stats_2d, stats_disp))
        return ds_list

    train_ds = ConcatDataset(create_datasets_from_indices(train_idx))
    val_ds = ConcatDataset(create_datasets_from_indices(val_idx))
    test_ds = ConcatDataset(create_datasets_from_indices(test_idx))
    
    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- 5. 模型与策略 ---
    model = MarkerDisplacementPredictor(num_points=num_points).to(device)
    
    criterion_base = nn.L1Loss()
    criterion_topk = TopKL1Loss(top_k_ratio=TOP_K_RATIO)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 6. 训练循环 ---
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_accuracy_by_threshold = {f'< {t:.2f}mm': [] for t in accuracy_thresholds_global}
    val_accuracy_by_threshold = {f'< {t:.2f}mm': [] for t in accuracy_thresholds_global}
    
    pbar = tqdm(range(NUM_EPOCHS), desc='Training', ncols=140)
    best_val_mae = float('inf')
    
    for epoch in pbar:
        use_topk = (epoch >= TOP_K_EPOCH)
        criterion = criterion_topk if use_topk else criterion_base
        
        # Train
        model.train()
        run_loss, cnt = 0.0, 0
        train_preds, train_targets = [], []
        
        for orig, curr, target_disp in train_loader:
            orig, curr, target_disp = orig.to(device), curr.to(device), target_disp.to(device)
            optimizer.zero_grad()
            pred = model(orig, curr)
            loss = criterion(pred, target_disp)
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
        
        if train_preds:
            tp_np = np.concatenate(train_preds)
            tt_np = np.concatenate(train_targets)
            train_maes.append(np.mean(np.abs(tp_np - tt_np)))
            t_acc = calculate_accuracy(tp_np, tt_np, accuracy_thresholds_global)
            for k,v in t_acc.items(): train_accuracy_by_threshold[k].append(v)
        
        # Validation
        model.eval()
        val_loss, val_cnt = 0.0, 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for orig, curr, target_disp in val_loader:
                orig, curr, target_disp = orig.to(device), curr.to(device), target_disp.to(device)
                pred = model(orig, curr)
                loss = criterion(pred, target_disp)
                val_loss += loss.item()
                val_cnt += 1
                
                p_real = denormalize_3d(pred, stats_disp)
                t_real = denormalize_3d(target_disp, stats_disp)
                val_preds.append(p_real.cpu().numpy())
                val_targets.append(t_real.cpu().numpy())
        
        val_losses.append(val_loss/val_cnt)
        
        if val_preds:
            vp_np = np.concatenate(val_preds)
            vt_np = np.concatenate(val_targets)
            avg_val_mae = np.mean(np.abs(vp_np - vt_np))
            val_maes.append(avg_val_mae)
            v_acc = calculate_accuracy(vp_np, vt_np, accuracy_thresholds_global)
            for k,v in v_acc.items(): val_accuracy_by_threshold[k].append(v)
            
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), 'best_model_pos_topk.pth')
        
        loss_name = 'TopK' if use_topk else 'L1'
        pbar.set_postfix({
            'L': loss_name,
            'TL': f'{train_losses[-1]:.4f}',
            'VL': f'{val_losses[-1]:.4f}',
            'VMae': f'{val_maes[-1]:.3f}',
            'Acc<0.05': f"{val_accuracy_by_threshold['< 0.05mm'][-1]:.1f}%"
        })

    print("\n===== FINAL EVALUATION (TEST SET) =====")
    model.load_state_dict(torch.load('best_model_pos_topk.pth', map_location=device))
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

    # --- 7. 训练过程可视化 ---
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
        
        # 5. Loss Contrast (Log Scale) - 用于观察后期微小收敛
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
        plt.suptitle(f"Training Metrics (PosEnc + Top-K L1)", fontsize=16, y=1.02)
        plt.savefig('training_metrics.png')
        print("Metrics plot saved to training_metrics.png")
        plt.show()
    
    # 保存配置
    torch.save({
        'num_points': num_points,
        'model_type': 'Transformer_TopKL1',
        'stats_2d': stats_2d,
        'stats_disp': stats_disp
    }, 'model_config_pos_topk.pth')
    
    print("Script execution finished.")