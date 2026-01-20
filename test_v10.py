import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

# 从net_train_v10.py导入必要的类和函数
from net_train_v10 import MarkerDisplacementDataset, MarkerDisplacementPredictor, denormalize_3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 位移阈值和准确率阈值
DISPLACEMENT_THRESHOLDS = [0.1, 0.3, 0.5, 1.0]  # 位移大于d的点
ACCURACY_THRESHOLDS = [0.05, 0.1, 0.3]  # 准确率阈值（mm）

def load_model_and_config(model_path, config_path, num_points):
    """
    加载模型和配置
    
    Args:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
        num_points: 每个样本的点数
    
    Returns:
        model: 加载的模型
        config: 配置字典（包含归一化统计信息）
    """
    # 加载模型
    model = MarkerDisplacementPredictor(num_points=num_points).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except RuntimeError as e:
        print(f"Loading Error: {e}")
        # 如果权重不匹配，尝试宽松加载 (虽然结构改变后应该重新训练，不应混用权重)
        print("Trying strict=False load...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        
    model.eval()
    
    # 加载配置
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    print(f"模型已加载: {model_path}")
    print(f"配置已加载: {config_path}")
    
    return model, config


def calculate_accuracy_by_displacement(predicted_disp, target_disp, 
                                       displacement_thresholds, accuracy_thresholds):
    """
    计算对于位移大于d的marker点的准确率
    
    Args:
        predicted_disp: 预测的位移 (N, 3) 或 (B, N, 3)
        target_disp: 真实的位移 (N, 3) 或 (B, N, 3)
        displacement_thresholds: 位移阈值列表 [d1, d2, ...]
        accuracy_thresholds: 准确率阈值列表 [t1, t2, ...]
    
    Returns:
        results: 字典，键为 'disp>d_acc<t'，值为准确率百分比
    """
    # 确保是2D数组
    if predicted_disp.ndim == 3:
        predicted_disp = predicted_disp.reshape(-1, 3)
        target_disp = target_disp.reshape(-1, 3)
    
    # 计算真实位移的模长
    true_displacement_magnitude = np.linalg.norm(target_disp, axis=1)

    # 计算预测误差（位移预测误差）
    pred_error = np.linalg.norm(predicted_disp - target_disp, axis=1)
    
    results = {}
    
    for d_thresh in displacement_thresholds:
        # 筛选位移大于d的点
        mask = true_displacement_magnitude > d_thresh
        filtered_points = np.sum(mask)
        
        if filtered_points == 0:
            # 没有符合条件的点
            for a_thresh in accuracy_thresholds:
                key = f'disp>{d_thresh:.1f}_acc<{a_thresh:.2f}'
                results[key] = 0.0
            continue
        
        # 对于筛选出的点，计算准确率
        filtered_errors = pred_error[mask]
        
        for a_thresh in accuracy_thresholds:
            # 误差小于准确率阈值的点数量
            correct = np.sum(filtered_errors < a_thresh)
            accuracy = (correct / filtered_points) * 100.0
            key = f'disp>{d_thresh:.1f}_acc<{a_thresh:.2f}'
            results[key] = accuracy
    
    return results


def evaluate_dataset(model, dataset, stats_2d, stats_disp, original_markers_3d,
                     displacement_thresholds, accuracy_thresholds, dataset_name=''):
    """
    在数据集上评估模型，计算按位移阈值筛选的准确率
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        stats_2d: 2D归一化统计信息
        stats_disp: 位移归一化统计信息
        original_markers_3d: 初始3D坐标 (N, 3)
        displacement_thresholds: 位移阈值列表
        accuracy_thresholds: 准确率阈值列表
        dataset_name: 数据集名称（用于打印）
    
    Returns:
        accuracy_results: 准确率结果字典
        all_pred_disp: 所有预测的位移
        all_target_disp: 所有真实的位移
    """
    model.eval()
    all_pred_disp = []
    all_target_disp = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for original_2d, deformed_2d, target_disp in dataloader:
            original_2d = original_2d.to(device)
            deformed_2d = deformed_2d.to(device)
            target_disp = target_disp.to(device)
            
            # 模型预测位移
            pred_disp = model(original_2d, deformed_2d)
            
            # 反归一化
            pred_disp_real = denormalize_3d(pred_disp, stats_disp)
            target_disp_real = denormalize_3d(target_disp, stats_disp)
            
            all_pred_disp.append(pred_disp_real.cpu().numpy())
            all_target_disp.append(target_disp_real.cpu().numpy())
    
    # 合并所有样本
    all_pred_disp = np.concatenate(all_pred_disp, axis=0).reshape(-1, 3)
    all_target_disp = np.concatenate(all_target_disp, axis=0).reshape(-1, 3)
    
    # 计算准确率
    accuracy_results = calculate_accuracy_by_displacement(
        all_pred_disp, all_target_disp,
        displacement_thresholds, accuracy_thresholds
    )
    
    # 打印结果
    if dataset_name:
        print(f"\n{dataset_name} 数据集结果:")
        print("=" * 80)
        for d_thresh in displacement_thresholds:
            print(f"\n位移 > {d_thresh:.1f}mm 的点:")
            for a_thresh in accuracy_thresholds:
                key = f'disp>{d_thresh:.1f}_acc<{a_thresh:.2f}'
                acc = accuracy_results.get(key, 0.0)
                print(f"  准确率 (<{a_thresh:.2f}mm): {acc:.2f}%")
        print("=" * 80)
    
    return accuracy_results, all_pred_disp, all_target_disp


def visualize_accuracy_by_displacement(train_results, test_results,
                                       displacement_thresholds, accuracy_thresholds,
                                       save_dir='test_results'):
    """
    可视化train和test数据集上按位移阈值的准确率
    
    Args:
        train_results: train数据集的准确率结果
        test_results: test数据集的准确率结果
        displacement_thresholds: 位移阈值列表
        accuracy_thresholds: 准确率阈值列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个准确率阈值创建一个图
    for a_thresh in accuracy_thresholds:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_accs = []
        test_accs = []
        disp_labels = []
        
        for d_thresh in displacement_thresholds:
            key = f'disp>{d_thresh:.1f}_acc<{a_thresh:.2f}'
            train_acc = train_results.get(key, 0.0)
            test_acc = test_results.get(key, 0.0)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            disp_labels.append(f'>{d_thresh:.1f}mm')
        
        x = np.arange(len(displacement_thresholds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_accs, width, label='Train', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8, color='coral')
        
        ax.set_xlabel('位移阈值', fontsize=12)
        ax.set_ylabel(f'准确率 (%) (误差 < {a_thresh:.2f}mm)', fontsize=12)
        ax.set_title(f'不同位移阈值下的准确率 (误差阈值: {a_thresh:.2f}mm)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(disp_labels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'accuracy_by_displacement_thresh{a_thresh:.2f}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率图已保存: {save_path}")
        plt.show()
    
    # 创建一个综合热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (results, title) in enumerate(zip([train_results, test_results], ['Train', 'Test'])):
        ax = axes[idx]

        # 构建矩阵
        matrix = []
        for d_thresh in displacement_thresholds:
            row = []
            for a_thresh in accuracy_thresholds:
                key = f'disp>{d_thresh:.1f}_acc<{a_thresh:.2f}'
                row.append(results.get(key, 0.0))
            matrix.append(row)
        matrix = np.array(matrix)
        
        # 绘制热力图
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        # 设置标签
        ax.set_xticks(np.arange(len(accuracy_thresholds)))
        ax.set_xticklabels([f'<{t:.2f}mm' for t in accuracy_thresholds])
        ax.set_yticks(np.arange(len(displacement_thresholds)))
        ax.set_yticklabels([f'>{d:.1f}mm' for d in displacement_thresholds])
        ax.set_title(f'{title} 准确率热力图', fontsize=12)
        
        # 添加数值标注
        for i in range(len(displacement_thresholds)):
            for j in range(len(accuracy_thresholds)):
                ax.text(j, i, f'{matrix[i, j]:.1f}%', ha="center", va="center", color="black", fontsize=10)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='准确率 (%)')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'accuracy_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"准确率热力图已保存: {save_path}")
    plt.show()


def visualize_2d_error_mapping(model, dataset, stats_2d, stats_disp, 
                                num_samples=12, save_dir='test_results'):
    
    """
    在train数据集上选择样本绘制2D误差映射图
    
    Args:
        model: 训练好的模型
        dataset: 数据集（train）
        stats_2d: 2D坐标归一化统计信息
        stats_disp: 位移归一化统计信息
        num_samples: 要可视化的样本数量
        save_dir: 保存图片的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择若干样本
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"警告: 请求的样本数超过数据集大小，使用全部 {total_samples} 个样本")
    
    subset_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # 直接访问选中的样本
    selected_samples = [dataset[i] for i in subset_indices]
    
    # 进行预测
    all_predicted_disp = []
    all_target_disp = []
    all_2d_inputs_norm = []
    
    model.eval()
    with torch.no_grad():
        for original_2d, deformed_2d, target_disp in selected_samples:
            # 添加batch维度并移到device
            original_2d = original_2d.unsqueeze(0).to(device)
            deformed_2d = deformed_2d.unsqueeze(0).to(device)
            target_disp = target_disp.unsqueeze(0).to(device)
            
            # 模型预测位移
            pred_disp = model(original_2d, deformed_2d)
            
            # 反归一化
            pred_disp_real = denormalize_3d(pred_disp, stats_disp)
            target_disp_real = denormalize_3d(target_disp, stats_disp)
            
            all_predicted_disp.append(pred_disp_real.cpu().numpy()[0])
            all_target_disp.append(target_disp_real.cpu().numpy()[0])
            # 保存归一化的2D输入
            all_2d_inputs_norm.append(deformed_2d.cpu().numpy()[0])
    
    # 计算每个样本的误差（位移预测误差）
    sample_errors = []
    for pred_disp, target_disp in zip(all_predicted_disp, all_target_disp):
        errors = np.linalg.norm(pred_disp - target_disp, axis=1)
        sample_errors.append({
            'mean': np.mean(errors),
            'max': np.max(errors),
            'median': np.median(errors)
        })
    
    # 创建可视化
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if num_samples == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (sample_idx, pred_disp, target_disp, input_2d_norm, error_info) in enumerate(zip(
        subset_indices, all_predicted_disp, all_target_disp, all_2d_inputs_norm, sample_errors)):
        
        ax = axes[idx]
        
        # 反归一化2D输入
        if stats_2d is not None:
            input_2d_denorm = input_2d_norm * stats_2d[1] + stats_2d[0]
        else:
            input_2d_denorm = input_2d_norm
        
        # 计算每个点的误差（位移预测误差）
        point_errors = np.linalg.norm(pred_disp - target_disp, axis=1)
        
        # 在2D输入平面上绘制，颜色表示位移预测误差
        scatter = ax.scatter(input_2d_denorm[:, 0], input_2d_denorm[:, 1], 
                           c=point_errors, cmap='YlOrRd', s=50, alpha=0.7,
                           edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('u (像素)', fontsize=10)
        ax.set_ylabel('v (像素)', fontsize=10)
        ax.set_xlim(0, 168)
        ax.set_ylim(0, 168)
        ax.set_title(f'样本 {sample_idx}\n平均误差: {error_info["mean"]:.3f}mm', fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='位移预测误差 (mm)', shrink=0.8)
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)): axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Train数据集 - 各marker位移预测误差', fontsize=16, y=1.02)
    
    save_path = os.path.join(save_dir, 'train_2d_error_mapping.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D误差映射图已保存到: {save_path}")
    plt.show()

    # 打印样本统计信息
    print("\n" + "="*60)
    print("样本预测统计信息")
    print("="*60)
    for idx, (sample_idx, error_info) in enumerate(zip(subset_indices, sample_errors)):
        print(f"\n样本 {sample_idx}:")
        print(f"  平均误差: {error_info['mean']:.4f} mm")
        print(f"  中位数误差: {error_info['median']:.4f} mm")
        print(f"  最大误差: {error_info['max']:.4f} mm")
    print("="*60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 配置参数
    marker_2d_dir = 'marker_2D'
    marker_3d_dir = 'marker_3D'

    model_path = 'best_model_pos_topk.pth'
    config_path = 'model_config_pos_topk.pth'
    
    if not os.path.exists(model_path):
        possible_models = glob.glob('*model*.pth')
        if possible_models:
            model_path = possible_models[0]
            print(f"未找到最佳模型，使用: {model_path}")
        else:
            print("错误: 找不到模型文件！")
            exit(1)

    if not os.path.exists(config_path):
        possible_configs = glob.glob('*config*.pth')
        if possible_configs:
            config_path = possible_configs[0]
            print(f"未找到指定配置，使用: {config_path}")
        else:
            print("错误: 找不到配置文件！")
            exit(1)
    
    try:
        config = torch.load(config_path, map_location=device, weights_only=False)
        num_points = config.get('num_points', 80)
        stats_2d = config.get('stats_2d', None)
        stats_disp = config.get('stats_disp', None)  # net_train_v7使用stats_disp
        model_type = config.get('model_type', None)
        print(f"从配置读取: num_points={num_points}")
        if model_type is not None:
            print(f"模型类型: {model_type}")
        if stats_2d is not None:
            print(f"2D归一化统计: mean={stats_2d[0]}, std={stats_2d[1]}")
        if stats_disp is not None:
            print(f"位移归一化统计: mean={stats_disp[0]}, std={stats_disp[1]}")
    except Exception as e:
        print(f"加载配置时出错: {e}")
        num_points = 80
        stats_2d = None
        stats_disp = None

    # 加载模型
    model, config = load_model_and_config(model_path, config_path, num_points)
    
    if stats_2d is None or stats_disp is None:
        print("警告: 配置文件中缺少归一化统计信息，将不使用归一化")
    else:
        print("\n注意: net_train_v10.py使用数据归一化，预测结果需要反归一化")
    
    print("\n加载初始点数据...")
    original_2d_path = 'markers_points_2d.csv'
    original_3d_path = 'marker_points.csv'
    
    if not os.path.exists(original_2d_path):
        print(f"错误: 找不到初始点2D坐标文件: {original_2d_path}")
        exit(1)
    if not os.path.exists(original_3d_path):
        print(f"错误: 找不到初始点3D坐标文件: {original_3d_path}")
        exit(1)
    
    print(f"加载初始点2D坐标: {original_2d_path}")
    original_2d_df = pd.read_csv(original_2d_path)
    original_markers_2d = original_2d_df[['u', 'v']].values.astype(np.float32)
    
    print(f"加载初始点3D坐标: {original_3d_path}")
    original_3d_df = pd.read_csv(original_3d_path)
    if 'x' in original_3d_df.columns:
        original_markers_3d = original_3d_df[['x', 'y', 'z']].values.astype(np.float32)
    else:
        original_markers_3d = original_3d_df[['deformed_x', 'deformed_y', 'deformed_z']].values.astype(np.float32)
    
    print(f"初始点数量: {len(original_markers_2d)}")
    
    # 加载数据文件
    marker_2d_files = sorted(glob.glob(os.path.join(marker_2d_dir, '*.csv')))
    marker_3d_files = sorted(glob.glob(os.path.join(marker_3d_dir, '*.csv')))
    
    # 创建数据集列表
    datasets = []
    for file_2d, file_3d in zip(marker_2d_files, marker_3d_files):
        if os.path.basename(file_2d) == os.path.basename(file_3d):
            dataset = MarkerDisplacementDataset(
                file_2d, file_3d, 
                original_markers_2d=original_markers_2d,
                original_markers_3d=original_markers_3d,
                stats_2d=stats_2d,
                stats_disp=stats_disp
            )
            datasets.append(dataset)
    
    dataset_indices = list(range(len(datasets)))
    train_indices, temp_indices = train_test_split(
        dataset_indices,
        test_size=(1 - 0.7),
        random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(0.1 / (0.2 + 0.1)),
        random_state=42
    )
    
    train_datasets = [datasets[i] for i in train_indices]
    test_datasets = [datasets[i] for i in test_indices]
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    print("\n" + "="*40 + " Train Set " + "="*40)
    train_results, _, _ = evaluate_dataset(
        model, train_dataset, stats_2d, stats_disp, original_markers_3d,
        DISPLACEMENT_THRESHOLDS, ACCURACY_THRESHOLDS, dataset_name='Train'
    )
    
    print("\n" + "="*40 + " Test Set " + "="*40)
    test_results, _, _ = evaluate_dataset(
        model, test_dataset, stats_2d, stats_disp, original_markers_3d,
        DISPLACEMENT_THRESHOLDS, ACCURACY_THRESHOLDS, dataset_name='Test'
    )
    
    visualize_accuracy_by_displacement(
        train_results, test_results,
        DISPLACEMENT_THRESHOLDS, ACCURACY_THRESHOLDS,
        save_dir='test_results'
    )
    
    visualize_2d_error_mapping(
        model, train_dataset, stats_2d, stats_disp,
        num_samples=12,
        save_dir='test_results'
    )
    
    print("\n测试完成！")