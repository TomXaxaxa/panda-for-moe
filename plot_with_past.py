import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 3D绘图不再需要

# --- 1. 配置区域 ---
# 请在此处修改以定义您想进行的对比

# 定义要对比的模型。
# 格式为: {"自定义模型名称": "存放.npy文件的文件夹路径"}
MODELS_TO_COMPARE = {
    "panda256": "./predictions/train_past/encoder",
    # "Koopman": "./predictions/test_set/koopman_koopa",
    # "PatchTST": "./predictions/test_set/koopman_patchtst",
}

# 指定要进行可视化对比的系统名称 (与 .npy 文件名中的系统名对应)
# 例如，如果文件是 '1689573031_Lorenz_one_shot_pred.npy'，这里就填写 'Lorenz'
TARGET_SYSTEM_NAME = "RayleighBenard_Duffing" # !! 请修改为您需要对比的系统名称

# 指定要绘制的样本索引 (sample index)。使用固定索引可确保模型间对比的公平性。
SAMPLE_INDEX_TO_PLOT = 1

# 保存生成图像的文件夹
PLOT_OUTPUT_DIR = f"./plots/train_past/comparison_{TARGET_SYSTEM_NAME}"

# --- 结束配置 ---


def plot_2d_comparison(context_data, true_future_data, predictions_dict, system_name, output_dir, sample_idx):
    """
    在多个子图中绘制单一维度的真实值与多个模型的预测值对比。
    真实值由 "context" (历史) 和 "true_future" (未来真实值) 拼接而成。
    预测值 ("prediction") 会被绘制在 "context" 之后，并与 "true_future" 对比。
    在 context 和 prediction 的分界处会绘制一条垂直虚线。
    """
    # --- 1. 数据校验和准备 ---
    if context_data.ndim != 3:
        print(f"  [跳过2D绘图] 系统 '{system_name}' 的 context 数据维度不正确，应为3维。")
        return

    num_samples, context_len, num_channels = context_data.shape
    future_len = true_future_data.shape[1]

    if sample_idx >= num_samples:
        print(f"  [错误] 指定的样本索引 {sample_idx} 超出范围 (0-{num_samples-1})。")
        return

    # 拼接成完整的地面真实值 (Ground Truth)
    # full_true_data 形状: (num_samples, context_len + future_len, num_channels)
    full_true_data = np.concatenate((context_data, true_future_data), axis=1)

    # --- 2. 选择维度并设置绘图样式 ---
    if num_channels < 3:
        dim_indices = list(range(num_channels))
    else:
        # 随机选择最多3个维度进行可视化，保证可复现性
        random.seed(0) 
        dim_indices = sorted(random.sample(range(num_channels), 3))

    if not dim_indices:
        print(f"  [跳过2D绘图] 系统 '{system_name}' 没有可供可视化的维度。")
        return
        
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions_dict)))
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]

    # --- 3. 开始绘图 ---
    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(20, 5 * len(dim_indices)), sharex=True)
    if len(dim_indices) == 1:
        axes = [axes]

    for i, dim_index in enumerate(dim_indices):
        ax = axes[i]
        
        # 绘制完整的地面真实值 (Ground Truth)
        ax.plot(full_true_data[sample_idx, :, dim_index], label='Ground Truth (Context + Future)', color='black', linewidth=2.5, zorder=10, alpha=0.7)

        # 依次绘制每个模型的预测值 (只在未来的时间段)
        x_pred = np.arange(context_len, context_len + future_len)
        for model_idx, (model_name, pred_data) in enumerate(predictions_dict.items()):
            if pred_data.shape[1] != future_len:
                print(f"  [警告] 模型 '{model_name}' 的预测长度与真实未来长度不匹配，跳过此模型。")
                continue
            ax.plot(
                x_pred,
                pred_data[sample_idx, :, dim_index],
                label=model_name,
                color=colors[model_idx],
                linestyle=linestyles[model_idx % len(linestyles)],
                linewidth=1.8
            )
        
        # 在 context 和 prediction 之间画一条垂直分割线
        # x 坐标为 context_len - 0.5，使其位于最后一个 context 点和第一个 prediction 点之间
        boundary_line_pos = context_len - 0.5
        ax.axvline(x=boundary_line_pos, color='red', linestyle='--', linewidth=2, label='Context/Prediction Boundary', zorder=11)
        
        ax.set_title(f"System: {system_name} - Dimension {dim_index} Comparison")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)

        # 创建唯一的图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())


    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f'Multi-Model Univariate Comparison - {system_name}\n(Sample Index: {sample_idx})', fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- 4. 保存图像 ---
    save_path = os.path.join(output_dir, f"Multi-Model_{system_name}_2D_comparison_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - 2D对比图已保存至: {save_path}")


# def plot_3d_comparison(...):
#     """
#     根据用户需求，3D绘图功能已在此版本中被移除/注释。
#     """
#     pass


def compare_models_on_system(models_dict, system_name, output_dir, sample_idx):
    """
    主函数，用于加载指定系统的数据并为所有模型生成对比图。
    新版：加载 context, one_shot_true, 和 one_shot_pred 文件。
    """
    print(f"--- 开始为系统 '{system_name}' 生成多模型对比图 ---")
    os.makedirs(output_dir, exist_ok=True)

    context_data = None
    true_future_data = None
    predictions_data = {}
    
    # --- 1. 数据加载 ---
    # 假设所有模型共享相同的 context 和 true_future 文件，我们从第一个模型目录加载它们
    first_model_dir = list(models_dict.values())[0]
    
    # 加载 Context 数据
    context_files = glob.glob(os.path.join(first_model_dir, f"*_{system_name}_context.npy"))
    if not context_files:
        print(f"[错误] 在 '{first_model_dir}' 中找不到系统 '{system_name}' 的 'context' 文件。")
        return
    try:
        context_data = np.load(context_files[0])
        print(f"成功加载 Context 数据: {context_files[0]}")
    except Exception as e:
        print(f"[错误] 加载 Context 文件 '{context_files[0]}' 失败: {e}")
        return

    # 加载 One-Shot True (未来真实值) 数据
    true_files = glob.glob(os.path.join(first_model_dir, f"*_{system_name}_one_shot_true.npy"))
    if not true_files:
        print(f"[错误] 在 '{first_model_dir}' 中找不到系统 '{system_name}' 的 'one_shot_true' 文件。")
        return
    try:
        true_future_data = np.load(true_files[0])
        print(f"成功加载 Ground Truth (Future) 数据: {true_files[0]}")
    except Exception as e:
        print(f"[错误] 加载 Ground Truth (Future) 文件 '{true_files[0]}' 失败: {e}")
        return

    # 接着，为每个模型查找并加载对应的 'one_shot_pred' 文件
    for model_name, model_dir in models_dict.items():
        pred_files = glob.glob(os.path.join(model_dir, f"*_{system_name}_one_shot_pred.npy"))
        if pred_files:
            pred_filepath = pred_files[0]
            try:
                predictions_data[model_name] = np.load(pred_filepath)
                print(f"  - 成功加载模型 '{model_name}' 的预测数据: {pred_filepath}")
            except Exception as e:
                print(f"  - [警告] 加载模型 '{model_name}' 的预测文件 '{pred_filepath}' 失败: {e}")
        else:
            print(f"  - [警告] 在 '{model_dir}' 中未找到系统 '{system_name}' 的 'one_shot_pred' 文件，将跳过此模型。")

    if not predictions_data:
        print("[错误] 未能成功加载任何模型的预测数据。无法生成对比图。")
        return

    print(f"\n--- 2. 开始生成可视化图像 ---")
    # --- 3. 调用绘图函数 ---
    plot_2d_comparison(context_data, true_future_data, predictions_data, system_name, output_dir, sample_idx)
    # plot_3d_comparison(...) # 3D绘图已根据要求禁用

    print("\n所有对比图生成完毕！")
    print(f"所有图像已保存至: {output_dir}")


if __name__ == "__main__":
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