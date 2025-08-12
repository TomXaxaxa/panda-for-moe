import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_and_process_hidden_states(base_dir):
    """
    加载所有数据集的隐状态，并进行处理以提取表征向量。
    
    Args:
        base_dir (str): 存放数据集文件夹的根目录。
        
    Returns:
        tuple: (all_vectors, all_labels, dataset_names)
               - all_vectors: 包含所有样本表征向量的 numpy 数组。
               - all_labels: 对应每个向量的标签（0, 1, 2...）。
               - dataset_names: 数据集名称列表。
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"指定的目录不存在: {base_dir}")

    # 自动从目录结构中获取数据集名称
    try:
        dataset_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not dataset_names:
            raise FileNotFoundError(f"在 {base_dir} 中没有找到任何数据集子文件夹。")
        print(f"Successfully discovered the following datasets: {', '.join(dataset_names)}")
    except Exception as e:
        print(f"Error scanning dataset directories: {e}")
        return None, None, None

    all_vectors_list = []
    all_labels_list = []

    # 遍历每个数据集
    for i, name in enumerate(dataset_names):
        dataset_path = os.path.join(base_dir, name)
        file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pt')]
        
        print(f"\nProcessing dataset '{name}' ({len(file_paths)} files)...")
        
        dataset_vectors = []
        # 使用 tqdm 显示处理进度
        for file_path in tqdm(file_paths, desc=f"  - Loading {name}"):
            try:
                # 加载一个批次的隐状态
                hidden_batch = torch.load(file_path, map_location=torch.device('cpu'))

                # ==================== 关键处理步骤 (已调整) ====================
                # Transformer 保存的张量形状: [batch_size, num_channels, d_model]
                # (已在模型代码中对 token/patch 维度求过平均)
                # 现在我们只需要在 channel 维度 (dim=1) 上取平均，即可得到最终表征。
                # 处理后形状: [batch_size, d_model]
                pooled_vectors = hidden_batch.mean(dim=1)
                # =============================================================
                
                dataset_vectors.append(pooled_vectors.numpy())

            except Exception as e:
                print(f"Warning: Could not load or process file {file_path}: {e}")
                continue
        
        if dataset_vectors:
            all_vectors_list.append(np.concatenate(dataset_vectors, axis=0))
            num_samples = all_vectors_list[-1].shape[0]
            all_labels_list.append(np.full(num_samples, i))

    if not all_vectors_list:
        print("Error: Failed to load any data. Please check file paths and content.")
        return None, None, None

    # 将所有数据集合并成一个大的 numpy 数组
    final_vectors = np.concatenate(all_vectors_list, axis=0)
    final_labels = np.concatenate(all_labels_list, axis=0)
    
    print("\nAll data has been loaded and processed.")
    print(f"Total samples: {final_vectors.shape[0]}, Feature dimension: {final_vectors.shape[1]}")
    
    return final_vectors, final_labels, dataset_names

def visualize_with_tsne(vectors, labels, names):
    """
    使用 t-SNE 进行降维并可视化。
    
    Args:
        vectors (np.array): 高维数据。
        labels (np.array): 数据标签。
        names (list): 标签对应的名称。
    """
    print("\nStarting t-SNE dimensionality reduction... (this may take some time)")
    
    # 使用 max_iter 兼容新版 scikit-learn
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors)
    print("t-SNE reduction complete.")

    # 绘图
    plt.figure(figsize=(12, 10))
    # 使用 "magma" 色彩方案以区别于之前的图
    palette = sns.color_palette("magma", n_colors=len(names))

    for i, name in enumerate(names):
        indices = (labels == i)
        plt.scatter(vectors_2d[indices, 0], vectors_2d[indices, 1],
                    c=[palette[i]], label=name, alpha=0.7, s=50)

    plt.title('t-SNE Visualization of Transformer Hidden States', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Dataset Origin', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图片
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    # 更新保存文件名
    save_path = os.path.join(output_dir, "transformer_hidden_states_tsne.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")

    plt.show()


if __name__ == '__main__':
    # ==================== 目录修改 ====================
    # 定义包含所有数据集子文件夹的根目录
    DATA_BASE_DIR = './hidden_states/transformer/'
    # ===============================================
    
    # 1. 加载并处理数据
    vectors, labels, dataset_names = load_and_process_hidden_states(DATA_BASE_DIR)
    
    # 2. 如果数据成功加载，则进行可视化
    if vectors is not None and labels is not None:
        visualize_with_tsne(vectors, labels, dataset_names)