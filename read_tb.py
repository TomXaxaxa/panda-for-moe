import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import os

# --- 1. 配置您的模型日志信息 ---
# 请在这里填写您所有需要对比的模型的日志信息。
# 每一个模型是一个字典，包含：
# - 'name': 您为这个模型起的名字，会显示在图例中 (e.g., 'Model_A', 'ResNet50_lr_0.01')
# - 'path': 该模型 TensorBoard 日志文件所在的【目录】路径
model_logs = [
    {
        "name": "Original",
        "path": "./checkpoints/Mamba256*2_1024/logs"  # <-- ‼️ 修改为您的第一个模型的日志目录
    },
    {
        "name": "Mamba-MoE",
        "path": "./checkpoints/Mamba256*2_1024+encoder+Deepseek-MOE/logs"  # <-- ‼️ 修改为您的第二个模型的日志目录
    }
    # 如果有更多模型，请像下面这样继续添加
    # {
    #     "name": "Model_Delta",
    #     "path": "./path/to/your/fourth/log_dir/"
    # },
]

# --- 2. 读取和处理数据 ---
all_loss_data = []
print("开始读取和处理日志文件...")

for model_info in model_logs:
    model_name = model_info["name"]
    log_dir = model_info["path"]
    
    print(f"--> 正在处理模型: {model_name}，路径: {log_dir}")
    
    # 检查路径是否存在，以提供更友好的提示
    if not os.path.isdir(log_dir):
        print(f"    [警告] 目录不存在或不是一个有效的目录，跳过: {log_dir}")
        continue
    
    try:
        # 初始化读取器，tbparse 会自动找到目录下的 tfevents 文件
        reader = SummaryReader(log_dir)
        df = reader.scalars
        
        # 筛选出 'train/loss' 指标 (‼️ 请根据您自己的tag名称修改)
        loss_df = df[df['tag'] == 'train/loss'].copy()
        
        if loss_df.empty:
            print(f"    [警告] 在 {log_dir} 中未找到 'train/loss' 标签。")
            # 打印出所有可用的标签，方便您检查和修改
            print(f"    该日志文件中可用的标签有: {df['tag'].unique().tolist()}")
            continue
            
        # 添加一个新列来标识模型
        loss_df['model'] = model_name
        
        all_loss_data.append(loss_df)
        print(f"    成功处理 {len(loss_df)} 条 'train/loss' 数据。")
        
    except Exception as e:
        print(f"    [错误] 处理目录 {log_dir} 时发生错误: {e}")

# --- 3. 合并数据并进行可视化 ---
if all_loss_data:
    # 将列表中的所有 DataFrame 合并为一个
    combined_df = pd.concat(all_loss_data, ignore_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8)) # 加大画布尺寸

    # 使用 seaborn 绘制线图，并使用更清晰的调色板和线条样式
    sns.lineplot(data=combined_df, x='step', y='value', hue='model', palette='tab10', linewidth=2)

    # --- 4. 设置图表属性 (英文) ---
    plt.title('Training Loss Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Train Loss', fontsize=14)
    plt.legend(title='Model', fontsize=11, loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 自动调整x轴和y轴的范围
    plt.xlim(0)
    plt.ylim(bottom=0)
    
    plt.tight_layout() # 调整布局以防止标签重叠

    # --- 5. 保存图表 ---
    output_image_path = "train_loss_comparison(MoE).png"
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

    print(f"\n✅ 图表已成功保存到: {output_image_path}")

else:
    print("\n❌ 没有找到任何可用于绘图的数据。请再次检查 `model_logs` 中的路径配置以及日志文件中的标签名称是否正确。")