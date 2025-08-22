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
    print(f"正在为演示创建虚拟数据，模型: {models}...")
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


def analyze_and_plot(base_dir='eval_results/patchtst', models_to_plot=None):
    """
    分析指定目录中的实验结果并生成图表。

    Args:
        base_dir (str, optional): 包含模型结果的根目录。
                                  默认为 'eval_results/patchtst'。
        models_to_plot (list, optional): 一个字符串列表，指定要绘制哪些模型。
                                         如果为 None，则自动检测 `base_dir` 下的所有模型。
                                         默认为 None。
    """
    base_path = Path(base_dir)

    if models_to_plot:
        models = sorted(models_to_plot)
        print(f"将分析指定模型: {models}")
    else:
        print(f"正在自动检测 '{base_dir}' 下的模型...")
        try:
            models = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
            if not models:
                print(f"在 '{base_dir}' 中未找到任何模型目录。")
                raise FileNotFoundError
            print(f"检测到模型: {models}")
        except FileNotFoundError:
            print(f"错误：目录 '{base_dir}' 不存在或为空。")
            default_models_for_demo = ['Mamba256+MoE+SE', 'Mamba256+SE', 'Another_Model']
            create_dummy_data(base_path, default_models_for_demo)
            models = default_models_for_demo

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
                        
                        # --- 关键修改开始 ---
                        # 如果指标是 'mse' 或 'mae'，则排除前10%的极端值
                        if metric in ['mse', 'mae']:
                            # 计算要保留的行数（底部90%）
                            n_rows_to_keep = int(len(df) * 0.85)
                            
                            # 对值进行排序，并选择最小的90%
                            sorted_values = df[metric].sort_values()
                            mean_val = sorted_values.iloc[:n_rows_to_keep].mean()
                        else:
                            # 对于其他指标，正常计算均值
                            mean_val = df[metric].mean()
                        # --- 关键修改结束 ---
                            
                        all_results.append({
                            'model': model_name, 
                            'pred_steps': pred_steps,
                            'metric': metric,
                            'mean': mean_val  # 使用新计算的均值
                        })
                    else:
                        if pred_steps == sorted(model_path.glob('metrics_pred*.csv'))[0] and model_name == models[0]:
                            print(f"警告：指标 '{metric}' 在 {csv_file} 中未找到，将被跳过。")
            except Exception as e:
                print(f"处理文件 '{csv_file}' 时出错: {e}")

    if not all_results:
        print("没有加载任何数据。无法生成图表。")
        return

    results_df = pd.DataFrame(all_results)

    # --- 绘图部分 (未修改) ---
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("未找到 Times New Roman 字体，将使用默认 serif 字体。")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle('Model Performance Comparison (Mean Trend on Linear Scale)', fontsize=22, fontname='Times New Roman')

    colormap = plt.get_cmap('tab10') 
    color_map = {model: colormap(i) for i, model in enumerate(models)}
    
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
                x_pos, model_data['mean'],
                marker='o',
                linestyle='--',
                markersize=8,
                alpha=0.8,
                color=color_map.get(model_name, 'black'), 
                label=model_name if i == 0 else ""
            )

        ax.set_ylabel(metric.upper(), fontsize=14, fontname='Times New Roman')
        ax.set_title(f'Metric: {metric.upper()}', fontsize=16, fontname='Times New Roman')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_yscale('linear')

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
    
    plt.savefig("model_comparison_plot.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # --- 使用方法 ---

    # 示例1: 自动检测 'eval_results/patchtst' 目录下的所有模型
    # analyze_and_plot(base_dir='eval_results/patchtst')

    # 示例2: 只分析和绘制指定的两个模型
    models_to_run = ['panda256+encoder', 'panda256+encoder+FlashAttention']
    analyze_and_plot(base_dir='eval_results/patchtst', models_to_plot=models_to_run)
    
    # 示例3: 如果目录不存在，脚本将自动创建包含三个模型的虚拟数据并绘图
    # analyze_and_plot(base_dir='non_existent_dir/patchtst')