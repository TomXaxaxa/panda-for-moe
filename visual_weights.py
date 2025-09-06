# visualize_weights.py (修正版)

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np # 确保导入 numpy

def visualize_attention_weights(file_path, num_samples_to_show=16):
    """
    加载并可视化保存的注意力权重。
    
    Args:
        file_path (str): .pt 文件的路径。
        num_samples_to_show (int): 要在热力图上显示的最大样本/通道数量。
    """
    try:
        # 加载数据
        print(f"Loading attention weights from: {file_path}")
        weights_tensor = torch.load(file_path)
        print(f"Successfully loaded tensor with shape: {weights_tensor.shape}")
        
        num_memory_slots = weights_tensor.shape[1]
        
        # --- 1. 可视化部分样本的热力图 ---
        plt.figure(figsize=(20, 10))
        
        num_to_plot = min(num_samples_to_show, weights_tensor.shape[0])
        subset_weights = weights_tensor[:num_to_plot, :]
        
        sns.heatmap(subset_weights.numpy(), cmap='viridis', cbar=True)
        
        plt.title(f'Attention Weights for First {num_to_plot} Samples/Channels')
        plt.xlabel(f'Memory Slot Index (0-{num_memory_slots - 1})')
        plt.ylabel('Sample/Channel Index')
        plt.tight_layout()
        plt.savefig("attention_heatmap.png")
        print("Saved attention heatmap to attention_heatmap.png")
        plt.show()

        # --- 2. 可视化所有槽位的平均使用率 ---
        avg_usage = weights_tensor.mean(dim=0)
        
        plt.figure(figsize=(20, 5))
        
        # ==================== 修正部分开始 ====================
        # 之前的代码: avg_usage.numpy().plot(kind='bar', width=0.8) 是错误的
        # 正确的方式是使用 plt.bar()
        
        # 准备 x 和 y 轴数据
        x_positions = np.arange(num_memory_slots)
        y_values = avg_usage.numpy()
        
        plt.bar(x_positions, y_values, width=0.8)
        # ==================== 修正部分结束 ====================
        
        plt.title('Average Attention Weight Across All Memory Slots')
        plt.xlabel('Memory Slot Index')
        plt.ylabel('Average Attention Weight')
        
        # 为了清晰，可以只显示部分刻度
        ax = plt.gca()
        ticks_to_show = max(1, num_memory_slots // 25) # 动态调整刻度密度
        ax.set_xticks(ax.get_xticks()[::ticks_to_show])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        plt.tight_layout()
        plt.savefig("average_attention_usage.png")
        print("Saved average attention usage bar chart to average_attention_usage.png")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Memory Attention Weights for PatchTST.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the memory_attention_weights.pt file."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=96,
        help="Number of samples/channels to display on the heatmap."
    )
    
    args = parser.parse_args()
    
    visualize_attention_weights(args.file_path, num_samples_to_show=args.samples)