import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import os

# --- 1. 定义文件路径 ---
# 您下载的 Parquet 文件的路径
parquet_file_path = './data/huggingface/test_zeroshot.parquet' 

# 您希望保存文件的根目录
output_base_directory = './data/huggingface/test'

# --- 2. 读取 Parquet 文件 ---
print(f"正在从 '{parquet_file_path}' 读取数据...")
df = pd.read_parquet(parquet_file_path)
print("数据读取完毕。")

# --- 3. 遍历每一行并按原始目录结构创建 .arrow 文件 ---
print("开始处理数据并生成 .arrow 文件...")
total_files = len(df)
for index, row in df.iterrows():
    # --- 核心逻辑修正 ---
    # 从 'source_directory' 列获取子文件夹名
    # 注意：请确保列名与您的文件完全一致，图片显示的是 'source_directory'
    subdirectory_name = row['_source_directory']
    
    # 从 'source_filename' 列获取原始文件名
    arrow_filename = row['_source_filename']
    
    # 构建完整的输出子文件夹路径
    target_subdirectory_path = os.path.join(output_base_directory, subdirectory_name)
    
    # 【关键步骤】确保这个子文件夹存在，如果不存在则创建它
    os.makedirs(target_subdirectory_path, exist_ok=True)
    
    # 构建最终要保存的完整文件路径
    output_file_path = os.path.join(target_subdirectory_path, arrow_filename)
    # --- 修正结束 ---

    # 创建一个只包含当前行数据的新 DataFrame
    row_df = pd.DataFrame([row])
    
    # 将其转换为 PyArrow Table
    table = pa.Table.from_pandas(row_df)
    
    # 将 Table 写入目标路径下的 .arrow 文件
    feather.write_feather(table, output_file_path)
    
    # 打印进度 (可选)
    if (index + 1) % 100 == 0:
        print(f"已处理 {index + 1}/{total_files}，最新保存至：{output_file_path}")

print(f"\n所有文件处理完成！")
print(f"拆分后的 .arrow 文件已按原始目录结构保存在 '{output_base_directory}' 文件夹中。")