import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_moe_report(file_path):
    """
    解析单个MoE报告文件，提取每个层中专家的使用百分比。

    Args:
        file_path (str): MoE报告文件的路径。

    Returns:
        tuple: 包含 (模块名列表, 专家数据字典) 的元组。
               专家数据字典的键是专家编号，值是该专家在各层的使用百分比列表。
               如果文件无法解析，则返回 (None, None)。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        print(f"错误: 无法读取文件 {file_path}. 原因: {e}")
        return None, None

    # 使用正则表达式查找所有MoE层的数据块
    layer_blocks = re.findall(r'--- MoE Gating Layer \(approx\. module (\d+)\) ---\n(.*?)(?=\n--- MoE Gating Layer|==================================================)', content, re.DOTALL)
    
    if not layer_blocks:
        print(f"警告: 在文件 {file_path} 中未找到MoE数据块。")
        return None, None

    module_names = []
    # 使用字典来存储每个专家的数据，例如 {'Expert 00': [36.68, 52.45, ...], 'Expert 01': [...]}
    expert_data = {} 

    for module_num, block_content in layer_blocks:
        module_names.append(f"Module {module_num}")
        
        # 提取当前块中每个专家的使用百分比
        expert_usages = re.findall(r'(Expert \d{2}):.*?\((\d+\.\d{2})%\)', block_content)
        
        if not expert_usages:
            print(f"警告: 在 Module {module_num} of {file_path} 中未找到专家使用数据。")
            continue

        for expert_name, percentage in expert_usages:
            # strip() 用于去除可能存在的多余空格
            expert_name = expert_name.strip()
            if expert_name not in expert_data:
                expert_data[expert_name] = []
            expert_data[expert_name].append(float(percentage))

    # 检查所有专家的列表长度是否一致
    list_lengths = {len(v) for v in expert_data.values()}
    if len(list_lengths) > 1:
        print(f"警告: 文件 {file_path} 中的数据不一致，某些专家在某些层中缺少数据。")
        # 进行数据补齐，用0填充缺失值
        max_len = max(list_lengths)
        for expert in expert_data:
            while len(expert_data[expert]) < max_len:
                expert_data[expert].append(0.0)

    return module_names, expert_data


def plot_expert_usage(module_names, expert_data, output_path):
    """
    使用并列柱状图可视化MoE专家的使用情况。

    Args:
        module_names (list): MoE模块名列表 (X轴标签)。
        expert_data (dict): 包含每个专家使用百分比的字典。
        output_path (str): 生成的图表图片的保存路径。
    """
    if not module_names or not expert_data:
        print(f"无有效数据可供绘图，跳过 {output_path}")
        return

    expert_labels = sorted(expert_data.keys())
    num_layers = len(module_names)
    num_experts = len(expert_labels)
    
    # 设置图形大小
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # 设置每个柱子的宽度
    bar_width = 0.8 / num_experts
    # 计算每个组的中心位置
    index = np.arange(num_layers)

    # 绘制每个专家的柱状图
    for i, expert_name in enumerate(expert_labels):
        percentages = expert_data[expert_name]
        # 计算每个专家柱状图的位置
        bar_positions = index - (num_experts / 2) * bar_width + i * bar_width + bar_width/2
        ax.bar(bar_positions, percentages, bar_width, label=expert_name)

    # 设置图表标题和标签
    ax.set_title(f'MoE Expert Usage Report\n({os.path.basename(output_path).replace(".png", ".txt")})', fontsize=18, fontweight='bold')
    ax.set_xlabel('MoE Gating Layer', fontsize=14)
    ax.set_ylabel('Usage Percentage (%)', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(module_names, rotation=45, ha="right")
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Experts', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 调整布局以防止标签重叠
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # 调整右边距为图例留出空间
    
    # 保存图表
    try:
        plt.savefig(output_path, dpi=300)
        print(f"图表已成功保存到: {output_path}")
    except IOError as e:
        print(f"错误: 无法保存图表到 {output_path}. 原因: {e}")
    
    # 关闭图形，释放内存
    plt.close(fig)


def process_directory(input_dir, output_dir):
    """
    递归处理输入目录中的所有.txt文件，并生成可视化图表。

    Args:
        input_dir (str): 包含MoE报告的输入目录。
        output_dir (str): 保存生成图表的输出目录。
    """
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    # 如果输出目录不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)

    # 递归遍历目录
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                print(f"\n--- 正在处理文件: {file_path} ---")
                
                # 解析报告
                module_names, expert_data = parse_moe_report(file_path)
                
                if module_names and expert_data:
                    # 构建输出文件名
                    output_filename = os.path.splitext(filename)[0] + '.png'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 绘制并保存图表
                    plot_expert_usage(module_names, expert_data, output_path)

if __name__ == '__main__':
    # --- 配置 ---
    # 将 'moe_report' 替换为你的报告所在的文件夹名称
    INPUT_FOLDER = './eval_results/moe report'
    # 生成的图表将保存在这个文件夹中
    OUTPUT_FOLDER = 'moe_report_charts'
    
    process_directory(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\n所有报告处理完毕。")

