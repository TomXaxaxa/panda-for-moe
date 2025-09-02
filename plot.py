import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D绘图所需

# --- 1. 配置区域 ---
# 请在此处修改以定义您想进行的对比

# 定义要对比的模型。
# 格式为: {"自定义模型名称": "存放.npy文件的文件夹路径"}
MODELS_TO_COMPARE = {
    "panda256": "./predictions/test_set/encoder",
    "prompt1": "./predictions/test_set/prompt_1",
    "prompt2": "./predictions/test_set/prompt_2",
}

# 指定要进行可视化对比的系统名称 (与 .npy 文件名中的系统名对应)
# 例如，如果文件是 '1689573031_Lorenz_preds.npy'，这里就填写 'Lorenz'
TARGET_SYSTEM_NAME = "ForcedFitzHughNagumo_CaTwoPlus" # !! 请修改为您需要对比的系统名称

# 指定要绘制的样本索引 (sample index)。使用固定索引可确保模型间对比的公平性。
SAMPLE_INDEX_TO_PLOT = 1

# 保存生成图像的文件夹
PLOT_OUTPUT_DIR = f"./plots/test_set/comparison_{TARGET_SYSTEM_NAME}"

# --- 结束配置 ---


def plot_2d_comparison(true_data, predictions_dict, system_name, output_dir, sample_idx):
    """
    在多个子图中绘制单一维度的真实值与多个模型的预测值对比。
    为了清晰地区分，真实值曲线会更突出。
    """
    # 检查数据维度
    if true_data.ndim != 3:
        print(f"  [跳过2D绘图] 系统 '{system_name}' 的数据维度不正确，应为3维。")
        return

    num_samples, _, num_channels = true_data.shape

    # 检查样本索引是否有效
    if sample_idx >= num_samples:
        print(f"  [错误] 指定的样本索引 {sample_idx} 超出范围 (0-{num_samples-1})。")
        return

    # 随机选择最多3个维度进行可视化
    if num_channels < 3:
        dim_indices = list(range(num_channels))
    else:
        dim_indices = sorted(random.sample(range(num_channels), 3))

    if not dim_indices:
        print(f"  [跳过2D绘图] 系统 '{system_name}' 没有可供可视化的维度。")
        return

    # --- 可视化增强：为不同模型定义清晰的颜色和线型 ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions_dict)))
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))] # 多种虚线样式

    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(18, 5 * len(dim_indices)), sharex=True)
    if len(dim_indices) == 1:
        axes = [axes]

    # 绘制每个维度的子图
    for i, dim_index in enumerate(dim_indices):
        ax = axes[i]
        
        # 优先绘制地面真实值 (Ground Truth)，使用更突出、易于辨识的样式
        ax.plot(true_data[sample_idx, :, dim_index], label='Ground Truth', color='black', linewidth=2.5, zorder=10, alpha=0.3)

        # 依次绘制每个模型的预测值
        for model_idx, (model_name, pred_data) in enumerate(predictions_dict.items()):
            ax.plot(
                pred_data[sample_idx, :, dim_index],
                label=model_name,
                color=colors[model_idx],
                linestyle=linestyles[model_idx % len(linestyles)],
                linewidth=1.8
            )
        
        ax.set_title(f"System: {system_name} - Dimension {dim_index} Comparison")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f'Multi-Model Univariate Comparison - {system_name}\n(Sample Index: {sample_idx})', fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 保存图像
    save_path = os.path.join(output_dir, f"Multi-Model_{system_name}_2D_comparison_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - 2D对比图已保存至: {save_path}")


def plot_3d_comparison(true_data, predictions_dict, system_name, output_dir, sample_idx):
    """
    在3D相空间中绘制真实轨迹与多个模型的预测轨迹。
    """
    # 检查维度是否足够进行3D绘图
    if true_data.shape[-1] < 3:
        print(f"  [跳过3D绘图] 系统 '{system_name}' 的维度 ({true_data.shape[-1]}) 小于3。")
        return

    # --- 可视化增强：定义颜色 ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions_dict)))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 提取用于绘图的数据
    true_traj = true_data[sample_idx, :, :3]
    
    # 优先绘制地面真实值轨迹 (Ground Truth)
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], label='Ground Truth Trajectory', color='black', linewidth=2.5, alpha=0.9, zorder=10)
    
    # 依次绘制每个模型的预测轨迹
    for i, (model_name, pred_data) in enumerate(predictions_dict.items()):
        pred_traj = pred_data[sample_idx, :, :3]
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], label=f'{model_name} Prediction', color=colors[i], linestyle='--', alpha=0.8)

    # 标记起点
    ax.scatter(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2], color='red', s=150, label='Start Point', marker='o', zorder=11, edgecolors='black')

    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_zlabel("Dimension 2")
    ax.set_title(f'Multi-Model 3D Phase Space Comparison - {system_name}\n(Sample Index: {sample_idx})', fontsize=16)
    ax.legend()
    ax.grid(True)

    # 保存图像
    save_path = os.path.join(output_dir, f"Multi-Model_{system_name}_3D_comparison_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - 3D对比图已保存至: {save_path}")


def compare_models_on_system(models_dict, system_name, output_dir, sample_idx):
    """
    主函数，用于加载指定系统的数据并为所有模型生成对比图。
    """
    print(f"--- 开始为系统 '{system_name}' 生成多模型对比图 ---")
    os.makedirs(output_dir, exist_ok=True)

    ground_truth_data = None
    predictions_data = {}
    
    # --- 1. 数据加载 ---
    # 首先，找到并加载 Ground Truth 文件 (我们假设所有模型目录中的 'trues' 文件是相同的)
    first_model_dir = list(models_dict.values())[0]
    true_files = glob.glob(os.path.join(first_model_dir, f"*_{system_name}_trues.npy"))
    
    if not true_files:
        print(f"[错误] 在 '{first_model_dir}' 中找不到系统 '{system_name}' 的 'trues' 文件。请检查路径和系统名称。")
        return
    
    true_filepath = true_files[0]
    try:
        ground_truth_data = np.load(true_filepath)
        print(f"成功加载 Ground Truth 数据: {true_filepath}")
    except Exception as e:
        print(f"[错误] 加载 Ground Truth 文件 '{true_filepath}' 失败: {e}")
        return

    # 接着，为每个模型查找并加载对应的 'preds' 文件
    for model_name, model_dir in models_dict.items():
        pred_files = glob.glob(os.path.join(model_dir, f"*_{system_name}_preds.npy"))
        if pred_files:
            pred_filepath = pred_files[0]
            try:
                predictions_data[model_name] = np.load(pred_filepath)
                print(f"  - 成功加载模型 '{model_name}' 的预测数据: {pred_filepath}")
            except Exception as e:
                print(f"  - [警告] 加载模型 '{model_name}' 的预测文件 '{pred_filepath}' 失败: {e}")
        else:
            print(f"  - [警告] 在 '{model_dir}' 中未找到系统 '{system_name}' 的预测文件，将跳过此模型。")

    if not predictions_data:
        print("[错误] 未能成功加载任何模型的预测数据。无法生成对比图。")
        return

    print(f"\n--- 2. 开始生成可视化图像 ---")
    # --- 3. 调用绘图函数 ---
    plot_2d_comparison(ground_truth_data, predictions_data, system_name, output_dir, sample_idx)
    plot_3d_comparison(ground_truth_data, predictions_data, system_name, output_dir, sample_idx)

    print("\n所有对比图生成完毕！")
    print(f"所有图像已保存至: {output_dir}")


if __name__ == "__main__":
    # 检查是否定义了要对比的模型
    if not MODELS_TO_COMPARE:
        print("错误：请在 'MODELS_TO_COMPARE' 字典中至少定义一个模型及其路径。")
    elif not TARGET_SYSTEM_NAME:
         print("错误：请在 'TARGET_SYSTEM_NAME' 变量中指定要对比的系统名称。")
    else:
        compare_models_on_system(
            models_dict=MODELS_TO_COMPARE,
            system_name=TARGET_SYSTEM_NAME,
            output_dir=PLOT_OUTPUT_DIR,
            sample_idx=SAMPLE_INDEX_TO_PLOT
        )