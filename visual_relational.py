import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_relational_scores(
    scores_tensor: torch.Tensor, 
    batch_idx: int, 
    head_idx: int, 
    layer_idx: int, 
    channel_names: list = None
):
    """
    可视化【单个样本】的 relational_scores 矩阵的热力图。
    
    Args:
        scores_tensor (torch.Tensor): 形状为 (bsz, num_heads, num_channels, num_channels) 的张量。
        batch_idx (int): 要可视化的批次索引。
        head_idx (int): 要可视化的注意力头索引。
        layer_idx (int): 正在可视化的层索引（用于标题）。
        channel_names (list): 每个通道的名称，用于坐标轴标签。
    """
    # 检查索引是否越界
    if batch_idx >= scores_tensor.shape[0]:
        print(f"Error: batch_idx {batch_idx} is out of bounds for tensor with batch size {scores_tensor.shape[0]}")
        return
    if head_idx >= scores_tensor.shape[1]:
        print(f"Error: head_idx {head_idx} is out of bounds for tensor with {scores_tensor.shape[1]} heads")
        return

    # 提取要绘制的数据
    scores_to_plot = scores_tensor[batch_idx, head_idx].detach().cpu().numpy()
    
    num_channels = scores_to_plot.shape[0]
    
    if channel_names is None:
        channel_names = [f"Var_{i+1}" for i in range(num_channels)]
    
    save_path = f'./relational_scores_layer{layer_idx}_head{head_idx}_batch{batch_idx}.png'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        scores_to_plot, 
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=channel_names,
        yticklabels=channel_names
    )
    plt.title(f"Relational Scores (Layer: {layer_idx}, Head: {head_idx}, Batch: {batch_idx})", fontsize=16)
    plt.xlabel("Key Variables", fontsize=12)
    plt.ylabel("Query Variables", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")
    plt.show()

# --- 新增函数：用于绘制平均分数 ---
def plot_average_relational_scores(
    scores_tensor: torch.Tensor, 
    head_idx: int, 
    layer_idx: int, 
    channel_names: list = None
):
    """
    可视化【所有样本平均后】的 relational_scores 矩阵的热力图。
    
    Args:
        scores_tensor (torch.Tensor): 形状为 (bsz, num_heads, num_channels, num_channels) 的张量。
        head_idx (int): 要可视化的注意力头索引。
        layer_idx (int): 正在可视化的层索引（用于标题）。
        channel_names (list): 每个通道的名称，用于坐标轴标签。
    """
    # 检查索引是否越界
    if head_idx >= scores_tensor.shape[1]:
        print(f"Error: head_idx {head_idx} is out of bounds for tensor with {scores_tensor.shape[1]} heads")
        return

    # --- 核心修改：对 batch 维度（dim=0）取平均 ---
    # 1. 选择特定头的所有批次数据
    scores_for_head = scores_tensor[:, head_idx, :, :]
    # 2. 沿 batch 维度计算平均值
    average_scores = torch.mean(scores_for_head, dim=0)
    
    scores_to_plot = average_scores.detach().cpu().numpy()
    
    num_channels = scores_to_plot.shape[0]
    
    if channel_names is None:
        channel_names = [f"Var_{i+1}" for i in range(num_channels)]
    
    # 修改保存路径以反映是平均值图像
    save_path = f'./relational_scores_layer{layer_idx}_head{head_idx}_averaged.png'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        scores_to_plot, 
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=channel_names,
        yticklabels=channel_names
    )
    # 修改标题以反映是平均值图像
    plt.title(f"Average Relational Scores (Layer: {layer_idx}, Head: {head_idx}, Averaged over {scores_tensor.shape[0]} samples)", fontsize=16)
    plt.xlabel("Key Variables", fontsize=12)
    plt.ylabel("Query Variables", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize relational scores from a saved .pt file.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True, 
        help="Path to the relational_scores.pt file."
    )
    parser.add_argument(
        "--layer_idx", 
        type=int, 
        default=0, 
        help="Index of the encoder layer to visualize."
    )
    parser.add_argument(
        "--batch_idx", 
        type=int, 
        default=0, 
        help="Index of the sample in the batch to visualize. (Ignored if --average_batch is used)"
    )
    parser.add_argument(
        "--head_idx", 
        type=int, 
        default=0, 
        help="Index of the attention head to visualize."
    )
    # --- 新增命令行参数 ---
    parser.add_argument(
        "--average_batch",
        action="store_true",  # 当出现这个参数时，其值为 True
        help="If set, plot the average scores across the entire batch."
    )
    args = parser.parse_args()

    # --- 数据加载 ---
    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return

    print(f"Loading data from: {args.file_path}")
    try:
        captured_data = torch.load(args.file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- 数据验证 ---
    if not isinstance(captured_data, dict):
        print("Error: Expected saved data to be a dictionary mapping layer_idx to tensors.")
        return
        
    if args.layer_idx not in captured_data:
        print(f"Error: Layer index {args.layer_idx} not found in the saved data.")
        print(f"Available layers: {list(captured_data.keys())}")
        return
        
    scores_tensor = captured_data[args.layer_idx]
    print(f"Tensor shape for layer {args.layer_idx}: {scores_tensor.shape}")

    # --- 修改调用逻辑：根据参数选择调用哪个函数 ---
    if args.average_batch:
        print("Averaging scores across the batch...")
        plot_average_relational_scores(
            scores_tensor=scores_tensor,
            head_idx=args.head_idx,
            layer_idx=args.layer_idx
        )
    else:
        print(f"Plotting for single sample at batch_idx {args.batch_idx}...")
        plot_relational_scores(
            scores_tensor=scores_tensor,
            batch_idx=args.batch_idx,
            head_idx=args.head_idx,
            layer_idx=args.layer_idx
        )

if __name__ == "__main__":
    main()