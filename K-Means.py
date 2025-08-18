import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse

def perform_clustering(input_path: str, output_path: str, num_experts: int):
    """
    加载表征文件，执行K-Means聚类，并保存聚类中心。
    """
    print(f"正在从 '{input_path}' 加载令牌表征...")
    try:
        # 加载表征，确保映射到CPU以防万一
        representations = torch.load(input_path, map_location='cpu')
        print("表征加载成功。")
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_path}' 未找到。请先运行修改后的评估脚本。")
        return

    print(f"\n开始对 {len(representations)} 个层的表征进行K-Means聚类...")
    print(f"目标聚类数量 (专家数量): {num_experts}")

    all_centroids = {}

    for layer_idx, tokens_tensor in representations.items():
        print(f"\n--- 正在处理第 {layer_idx} 层 ---")
        print(f"  表征张量形状: {tokens_tensor.shape}")

        if tokens_tensor.shape[0] < num_experts:
            print(f"  警告：令牌数量 ({tokens_tensor.shape[0]}) 少于聚类数量 ({num_experts})。跳过该层。")
            continue
            
        # 转换为numpy以使用scikit-learn
        numpy_tokens = tokens_tensor.numpy()

        # 使用MiniBatchKMeans以提高效率和处理大规模数据
        kmeans = MiniBatchKMeans(
            n_clusters=num_experts,
            random_state=42,
            batch_size=4096,  # 可根据您的内存大小调整
            n_init='auto',
            verbose=0  # 设为1可以看到聚类过程
        )
        
        print("  正在拟合K-Means模型...")
        kmeans.fit(numpy_tokens)
        
        # 获取聚类中心并转换回torch.Tensor
        centroids_numpy = kmeans.cluster_centers_
        centroids_torch = torch.from_numpy(centroids_numpy).float()
        
        all_centroids[layer_idx] = centroids_torch
        print(f"  第 {layer_idx} 层聚类完成，聚类中心形状: {centroids_torch.shape}")

    # 保存所有聚类中心
    torch.save(all_centroids, output_path)
    print(f"\n所有层的聚类中心已成功保存到: '{output_path}'")
    
    # 验证
    print("\n--- 验证保存的文件 ---")
    loaded_data = torch.load(output_path)
    for idx, cents in loaded_data.items():
        print(f"  加载的第 {idx} 层中心形状: {cents.shape}")


if __name__ == "__main__":
    # 使用argparse来方便地从命令行指定参数
    parser = argparse.ArgumentParser(description="对提取的令牌表征执行K-Means聚类。")
    parser.add_argument(
        "--input_path",
        type=str,
        default="ffn_representations.pt",
        help="包含提取的令牌表征的输入文件路径。"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="panda_moe_centroids.pt",
        help="用于保存最终聚类中心的输出文件路径。"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="要生成的聚类中心数量（即专家数量）。"
    )
    
    args = parser.parse_args()
    
    perform_clustering(args.input_path, args.output_path, args.num_experts)