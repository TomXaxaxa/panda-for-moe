import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_process_data(base_dir, model_type):
    """
    根据模型类型加载并处理数据。
    
    Args:
        base_dir (str): 数据根目录。
        model_type (str): 'mamba' 或 'transformer'。
        
    Returns:
        tuple: (vectors, labels, dataset_names)
    """
    print("-" * 50)
    print(f"开始加载模型 '{model_type}' 的表征向量...")
    print(f"数据源: {base_dir}")

    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        return None, None, None

    try:
        dataset_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not dataset_names:
            raise FileNotFoundError(f"在 {base_dir} 中没有找到数据集子文件夹。")
    except Exception as e:
        print(f"扫描目录出错: {e}")
        return None, None, None

    all_vectors_list = []
    all_labels_list = []

    for i, name in enumerate(dataset_names):
        dataset_path = os.path.join(base_dir, name)
        file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pt')]
        
        dataset_vectors = []
        for file_path in tqdm(file_paths, desc=f"  - 加载 {name}"):
            hidden_batch = torch.load(file_path, map_location=torch.device('cpu'))
            
            # 根据模型类型应用不同的处理逻辑
            if model_type == 'mamba':
                # Mamba 存的是 [batch, channel, patch, d_model]，需两次平均
                pooled_vectors = hidden_batch.mean(dim=2).mean(dim=1)
            elif model_type == 'transformer':
                # Transformer 存的是 [batch, channel, d_model]，只需一次平均
                pooled_vectors = hidden_batch.mean(dim=1)
            else:
                raise ValueError("模型类型必须是 'mamba' 或 'transformer'")
                
            dataset_vectors.append(pooled_vectors.numpy())
        
        if dataset_vectors:
            all_vectors_list.append(np.concatenate(dataset_vectors, axis=0))
            num_samples = all_vectors_list[-1].shape[0]
            all_labels_list.append(np.full(num_samples, i))

    if not all_vectors_list:
        print("错误：未能加载任何数据。")
        return None, None, None

    final_vectors = np.concatenate(all_vectors_list, axis=0)
    final_labels = np.concatenate(all_labels_list, axis=0)
    
    print(f"'{model_type}' 数据加载完成。总样本数: {final_vectors.shape[0]}")
    return final_vectors, final_labels, dataset_names

def run_classification_experiment(X, y, model_name, dataset_names):
    """
    运行线性分类实验并报告结果。
    """
    print(f"\n为 '{model_name}' 模型运行分类实验...")
    
    # 1. 划分训练集和测试集 (70% 训练, 30% 测试)
    # stratify=y 确保训练集和测试集中各类别的比例与原始数据一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 2. 初始化并训练 RidgeClassifier
    print("正在训练 RidgeClassifier...")
    classifier = RidgeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # 3. 在测试集上进行预测
    y_pred = classifier.predict(X_test)

    # 4. 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=dataset_names, zero_division=0)
    
    print("\n--- 分类结果 ---")
    print(f"模型: {model_name}")
    print(f"分类准确率 (Accuracy): {accuracy:.4f}")
    print("\n详细分类报告:")
    print(report)
    
    return accuracy

if __name__ == '__main__':
    # 定义两个模型的数据目录
    MAMBA_DIR = './hidden_states/mamba/'
    TRANSFORMER_DIR = './hidden_states/transformer/'
    
    results = {}
    
    # --- 运行 Mamba 实验 ---
    mamba_X, mamba_y, mamba_names = load_and_process_data(MAMBA_DIR, 'mamba')
    if mamba_X is not None:
        mamba_accuracy = run_classification_experiment(mamba_X, mamba_y, 'Mamba', mamba_names)
        results['Mamba'] = mamba_accuracy

    # --- 运行 Transformer 实验 ---
    transformer_X, transformer_y, transformer_names = load_and_process_data(TRANSFORMER_DIR, 'transformer')
    if transformer_X is not None:
        transformer_accuracy = run_classification_experiment(transformer_X, transformer_y, 'Transformer', transformer_names)
        results['Transformer'] = transformer_accuracy
        
    # --- 最终结果对比 ---
    print("\n" + "="*50)
    print("          实验最终总结")
    print("="*50)
    if 'Mamba' in results and 'Transformer' in results:
        print(f"Mamba 表征空间可分性 (分类准确率):       {results['Mamba']:.2%}")
        print(f"Transformer 表征空间可分性 (分类准确率): {results['Transformer']:.2%}")
        
        diff = abs(results['Transformer'] - results['Mamba'])
        winner = 'Transformer' if results['Transformer'] > results['Mamba'] else 'Mamba'
        print(f"\n结论: '{winner}' 模型的表征空间具有显著更高的可分离性，准确率高出 {diff:.2%}")
    else:
        print("未能完成所有实验，无法进行最终对比。")
    print("="*50)