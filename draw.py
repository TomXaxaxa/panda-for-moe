import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_dummy_data(base_dir, models):
    """
    为测试目的创建虚拟文件结构和数据。
    如果您的真实数据已存在，可以跳过或注释掉对此函数的调用。
    """
    print("正在为演示创建虚拟数据...")
    pred_steps = [64, 128, 192, 256, 320, 384, 448, 512]
    
    for model in models:
        data_path = base_dir / model / 'test_example'
        data_path.mkdir(parents=True, exist_ok=True)
        for steps in pred_steps:
            file_path = data_path / f'metrics_pred{steps}.csv'
            
            # 模拟大部分数据在0-1，但有极端值
            normal_data_mse = np.random.rand(18) * 0.5 
            outlier_mse = np.array([1e5, 1e6]) 
            mse_values = np.concatenate((normal_data_mse, outlier_mse))
            np.random.shuffle(mse_values)

            normal_data_mae = np.random.rand(18) * 0.5
            outlier_mae = np.array([1e4, 1e5])
            mae_values = np.concatenate((normal_data_mae, outlier_mae))
            np.random.shuffle(mae_values)
            
            data = {
                'system': np.arange(20),
                'mse': mse_values,
                'mae': mae_values,
                'smape': np.random.rand(20), # SMAPE 通常在0-200之间
                'spearman': np.random.rand(20) * 0.5 + 0.2,
                'system_dims': np.random.randint(1, 5, 20),
                'n_system_samples': np.random.randint(100, 200, 20)
            }
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
    print("虚拟数据创建完成。")


def analyze_and_plot(base_dir='eval_results/patchtst'):
    """
    分析指定目录中的实验结果并生成图表。
    """
    base_path = Path(base_dir)

    models = [
        'Mamba256+MoE+SE',
        'Mamba256+SE'
    ]
    models.sort()

    if not base_path.exists():
        print(f"错误：目录 '{base_dir}' 不存在。正在创建虚拟数据...")
        # 为了更好地演示symlog，修改了虚拟数据生成
        create_dummy_data(base_path, models)

    metrics_to_plot = ['mse', 'mae', 'smape', 'spearman']
    
    all_results = []
    
    for model_name in models:
        model_path = base_path / model_name / 'test_example'
        if not model_path.exists():
            print(f"警告：目录 '{model_path}' 未找到，跳过 '{model_name}'。")
            continue
            
        for csv_file in sorted(model_path.glob('metrics_pred*.csv')):
            try:
                pred_steps_str = csv_file.stem.split('pred')[-1]
                pred_steps = int(pred_steps_str)
                df = pd.read_csv(csv_file)
                
                for metric in metrics_to_plot:
                    if metric in df.columns:
                        all_results.append({
                            'model': model_name, 
                            'pred_steps': pred_steps,
                            'metric': metric,
                            'median': df[metric].median()
                        })
                    else:
                        if pred_steps == 64 and model_name == models[0]:
                            print(f"警告：指标 '{metric}' 在 {csv_file} 中未找到，将被跳过。")
            except Exception as e:
                print(f"处理文件 '{csv_file}' 时出错: {e}")

    if not all_results:
        print("没有加载任何数据。无法生成图表。")
        return

    results_df = pd.DataFrame(all_results)

    # 绘图部分
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("未找到 Times New Roman 字体，将使用默认 serif 字体。")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    # --- 修改: 更新图表标题以反映symlog尺度 ---
    fig.suptitle('Model Performance Comparison (Median Trend on Symlog Scale)', fontsize=22, fontname='Times New Roman')

    colors = {
        'Mamba256+MoE+SE': '#1f77b4',
        'Mamba256+SE': '#ff7f0e',
    }
    
    unique_pred_steps = sorted(results_df['pred_steps'].unique())
    
    if len(unique_pred_steps) > 1:
        min_step_diff = np.min(np.diff(unique_pred_steps))
        point_gap = min_step_diff * 0.1
    else:
        point_gap = 8

    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes_flat[i]
        
        n_models = len(models)
        for j, model_name in enumerate(models):
            model_data = results_df[(results_df['model'] == model_name) & (results_df['metric'] == metric)]
            
            if model_data.empty:
                continue

            model_data = model_data.sort_values('pred_steps')
            
            offset = (j - (n_models - 1) / 2) * point_gap
            x_pos = model_data['pred_steps'] + offset
            
            ax.plot(
                x_pos, model_data['median'],
                marker='o',
                linestyle='--',
                markersize=8,
                alpha=0.8,
                color=colors.get(model_name, 'black'), 
                label=model_name if i == 0 else ""
            )

        ax.set_ylabel(metric.upper(), fontsize=14, fontname='Times New Roman')
        ax.set_title(f'Metric: {metric.upper()}', fontsize=16, fontname='Times New Roman')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 关键修改 ---
        # 将Y轴设置为对称对数坐标轴。
        # linthresh 定义了0点附近线性区域的大小，您可以根据数据分布调整此值。
        ax.set_yscale('symlog', linthresh=0.1)

    for ax in axes[1, :]:
        ax.set_xlabel('Prediction Steps', fontsize=14, fontname='Times New Roman')
    
    plt.xticks(unique_pred_steps)
    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_size(12)
    
    for i in range(len(metrics_to_plot), len(axes_flat)):
        axes_flat[i].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10, title='Models')
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    if legend.get_title():
        legend.get_title().set_fontname('Times New Roman')
        legend.get_title().set_fontsize('12')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # --- 修改: 保存为新文件名 ---
    plt.savefig("model_comparison_plot_MoE.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    analyze_and_plot(base_dir='eval_results/patchtst')