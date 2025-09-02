import os
import shutil
import pyarrow.ipc as ipc

# --- 配置路径 ---
# 源目录：包含所有测试数据集
SOURCE_DIR = './data/new_skew40/test'
# 目标目录：用于存放筛选出的数据集
DEST_DIR = './data/new_skew40/test_fixed_dims'

def find_arrow_file(folder_path):
    """在一个文件夹中查找第一个.arrow文件"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.arrow'):
            return os.path.join(folder_path, filename)
    return None

def main():
    """主函数，执行数据集筛选和复制任务"""
    
    # 确保源目录存在
    if not os.path.exists(SOURCE_DIR):
        print(f"错误：源目录 '{SOURCE_DIR}' 不存在。请检查路径。")
        return

    # 创建目标目录，如果它不存在
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"已创建目标目录：'{DEST_DIR}'")

    print("开始筛选数据集...")
    
    # 遍历源目录下的所有子文件夹（即数据集）
    for dataset_name in os.listdir(SOURCE_DIR):
        dataset_path = os.path.join(SOURCE_DIR, dataset_name)
        
        # 仅处理文件夹
        if not os.path.isdir(dataset_path):
            continue

        print(f"\n- 正在检查数据集：'{dataset_name}'")
        
        # 查找该数据集下的第一个 .arrow 文件
        arrow_file_path = find_arrow_file(dataset_path)

        if not arrow_file_path:
            print(f"  警告：在 '{dataset_name}' 中未找到 .arrow 文件，跳过。")
            continue
            
        try:
            # 打开 .arrow 文件以读取其形状信息
            with ipc.open_file(arrow_file_path) as reader:
                # 修正：使用 read_all() 一次性读取整个文件为一个 Table
                arrow_table = reader.read_all()
                
                # 获取 'target._np_shape' 列的第一条记录
                shape_column = arrow_table.column('target._np_shape')
                shape_value = shape_column[0].as_py()
                
                # 检查形状是否符合条件：
                if isinstance(shape_value, list) and len(shape_value) > 0 and shape_value[0] == 3:
                    print(f"  √ 形状符合要求 ({shape_value})，开始复制整个数据集...")
                    
                    # 复制整个数据集文件夹到目标目录
                    dest_dataset_path = os.path.join(DEST_DIR, dataset_name)
                    if os.path.exists(dest_dataset_path):
                        print(f"  目标文件夹 '{dest_dataset_path}' 已存在，先删除旧文件夹。")
                        shutil.rmtree(dest_dataset_path)
                    
                    shutil.copytree(dataset_path, dest_dataset_path)
                    print(f"  成功将 '{dataset_name}' 复制到 '{dest_dataset_path}'")
                else:
                    print(f"  × 形状不符合要求 ({shape_value})，跳过。")
                    
        except Exception as e:
            print(f"  错误：处理 '{dataset_name}' 时发生异常：{e}，跳过该数据集。")
            
    print("\n任务完成。")

if __name__ == "__main__":
    main()