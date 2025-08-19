import os
import glob
import numpy as np
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 配置区域 ---
# 请修改为您的数据所在的根目录
INPUT_DIRECTORY = './data/new_skew40/test/Thomas_RayleighBenard/'
# 请设置您希望保存结果图像的路径和文件名
OUTPUT_PLOT_PATH = './dft_spectrum.png' # 修改了文件名以防覆盖
# -----------------

def load_and_process_data(root_dir: str):
    """
    递归扫描目录，加载所有 .arrow 文件，并将它们合并成一个三维 NumPy 数组。
    """
    file_paths = glob.glob(os.path.join(root_dir, '**', '*.arrow'), recursive=True)
    if not file_paths:
        print(f"错误：在目录 '{root_dir}' 中没有找到任何 .arrow 文件。")
        return None
    print(f"找到了 {len(file_paths)} 个 .arrow 文件。开始处理...")
    all_samples = []
    for file_path in tqdm(file_paths, desc="正在读取文件"):
        try:
            table = feather.read_table(file_path)
            row = table.to_pandas().iloc[0]
            target_data = row['target']
            target_shape = row['target._np_shape']
            if len(target_shape) != 2:
                print(f"警告：文件 '{file_path}' 的形状不是二维，已跳过。")
                continue
            reshaped_data = np.array(target_data).reshape(target_shape)
            transposed_data = reshaped_data.T
            all_samples.append(transposed_data)
        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生错误: {e}")
    if not all_samples:
        print("未能成功加载任何数据。")
        return None
    final_array = np.stack(all_samples, axis=0)
    print("数据加载和整合完成！")
    print(f"最终数组尺寸: {final_array.shape}")
    return final_array


def perform_dft_and_plot(data_array: np.ndarray, output_path: str):
    """
    对时序数据进行DFT，并为每个维度绘制独立的、X轴范围为0-0.1的子图。
    """
    if data_array is None or data_array.ndim != 3:
        print("错误：输入数据无效，无法进行DFT。")
        return
    print("正在进行DFT变换...")
    fft_result = np.fft.fft(data_array, axis=1)
    n_samples, n_lengths, n_dims = data_array.shape
    magnitudes = np.abs(fft_result[:, :n_lengths // 2, :])
    avg_magnitudes = np.mean(magnitudes, axis=0)
    dt = 1 
    freqs = np.fft.fftfreq(n_lengths, d=dt)[:n_lengths // 2]
    print("正在生成频谱图...")

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    fig.suptitle('Average DFT Amplitude Spectrum by Dimension (Zoomed)', fontsize=16)
    
    for i, ax in enumerate(axes):
        ax.plot(freqs, avg_magnitudes[:, i], color=f'C{i}')
        ax.set_title(f'Dimension {i+1}')
        ax.set_ylabel('Average Magnitude')
        ax.set_ylim(0, 100)
        ax.grid(True)
        
    # 只在最下面的子图显示X轴标签
    axes[-1].set_xlabel('Frequency (cycles/sample)')
    
    # --- 新增代码在这里 ---
    # 设置所有子图的X轴显示范围为 0 到 0.1
    axes[-1].set_xlim(0.1, 0.5)
    # --------------------
    
    # 调整布局以防标题和标签重叠
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 保存图像
    try:
        plt.savefig(output_path, dpi=300)
        print(f"频谱图已成功保存到: '{output_path}'")
    except Exception as e:
        print(f"保存图像时发生错误: {e}")
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    time_series_data = load_and_process_data(INPUT_DIRECTORY)
    if time_series_data is not None:
        perform_dft_and_plot(time_series_data, OUTPUT_PLOT_PATH)