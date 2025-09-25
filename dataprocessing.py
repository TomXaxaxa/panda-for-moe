import torch
import time
import threading
import os # 引入 os 模块

# ==============================================================================
# --- 在这里修改配置 ---

# 模式选择: 'size' 或 'ratio'
# 'size'  -> 根据下面的 TARGET_SIZE_GB 来分配固定大小的显存
# 'ratio' -> 根据下面的 TARGET_RATIO 来分配一个比例的显存
MODE = 'size'

# 要使用的GPU设备索引。如果为空列表 `[]`，则表示使用所有可用的GPU。
# 示例: [0, 1, 3] 表示使用 GPU 0, 1, 3
GPUS_TO_USE = [3] 

# 当 MODE = 'size' 时生效: 指定要占用的显存大小（单位：GB）
TARGET_SIZE_GB = 45

# 当 MODE = 'ratio' 时生效: 指定要占用的显存比例（0.0 到 1.0 之间）
TARGET_RATIO = 0.9

# --- 配置修改结束 ---
# ==============================================================================


def allocate_and_run_on_device(device, all_tensors, target_mem_bytes):
    """
    在指定设备上分配目标大小的显存，并持续进行轻量运算。
    (此函数内部与上一版代码相同，此处为简洁省略，请使用上一版中的函数定义)
    """
    try:
        # --- 打印初始状态 ---
        print(f"[{device}] ----- 分配前显存状态 -----")
        total_mem = torch.cuda.get_device_properties(device).total_memory
        allocated_mem = torch.cuda.memory_allocated(device)
        print(f"[{device}] 总显存: {total_mem / 1024**3:.2f} GB")
        print(f"[{device}] （本进程）已分配显存: {allocated_mem / 1024**3:.2f} GB")

        # --- 安全检查 ---
        if target_mem_bytes >= total_mem:
            print(f"[{device}] 警告: 目标显存 ({target_mem_bytes / 1024**3:.2f} GB) 大于或等于总显存 ({total_mem / 1024**3:.2f} GB)。")
            target_mem_bytes = int(total_mem * 0.95)
            print(f"[{device}] 已将目标自动调整为总显存的95%: {target_mem_bytes / 1024**3:.2f} GB。")
        
        print(f"[{device}] 目标分配显存: {target_mem_bytes / 1024**3:.2f} GB")

        all_tensors[device] = []
        
        chunk_sizes = [1024*1024*1024, 128*1024*1024, 16*1024*1024, 1*1024*1024, 1024]

        print(f"\n[{device}] 正在尝试分配显存至目标值...")
        
        for chunk_size_bytes in chunk_sizes:
            tensor_elements = int(chunk_size_bytes / 4)
            if tensor_elements == 0: continue

            while True:
                if torch.cuda.memory_allocated(device) + chunk_size_bytes > target_mem_bytes:
                    break
                try:
                    new_tensor = torch.zeros(tensor_elements, dtype=torch.float32, device=device)
                    all_tensors[device].append(new_tensor)
                    current_allocated = torch.cuda.memory_allocated(device)
                    print(f"[{device}] 已分配显存: {current_allocated / 1024**3:.2f} GB", end='\r')
                except RuntimeError:
                    break
        
        print(f"\n\n[{device}] ----- 显存分配完成 -----")
        allocated_mem = torch.cuda.memory_allocated(device)
        reserved_mem = torch.cuda.memory_reserved(device)
        allocated_ratio = (allocated_mem / total_mem) * 100
        print(f"[{device}] 最终已分配显存: {allocated_mem / 1024**3:.2f} GB ({allocated_ratio:.2f}%)")
        print(f"[{device}] 最终已缓存显存: {reserved_mem / 1024**3:.2f} GB")
        
        compute_tensor_size = 1024
        compute_tensor = torch.rand(compute_tensor_size, compute_tensor_size, device=device)

        print(f"\n[{device}] 显存已占用。开始进行轻量运算以“伪装”成正常进程。")
        
        while True:
            _ = torch.matmul(compute_tensor, compute_tensor)
            time.sleep(1)
            # 为了终端更干净，可以将这个打印注释掉
            # print(f"[{device}] 仍在运行并计算...", end='\r')

    except Exception as e:
        print(f"\n[{device}] 运行时发生错误: {e}")

def main():
    if not torch.cuda.is_available():
        print("未检测到可用的CUDA设备。")
        return

    num_total_gpus = torch.cuda.device_count()
    print(f"系统共检测到 {num_total_gpus} 个CUDA设备。")
    
    # 根据配置决定要使用的GPU
    if not GPUS_TO_USE: # 如果列表为空
        gpu_indices_to_use = list(range(num_total_gpus))
        print("配置为空，将使用所有可用的GPU。")
    else:
        gpu_indices_to_use = GPUS_TO_USE
        print(f"根据配置，指定要使用的GPU: {gpu_indices_to_use}")

    # 验证用户输入的GPU索引是否有效
    for gpu_idx in gpu_indices_to_use:
        if gpu_idx < 0 or gpu_idx >= num_total_gpus:
            print(f"错误：配置的GPU索引 {gpu_idx} 无效。请检查。")
            return

    all_tensors = {}
    threads = []
    
    print("-" * 40)
    for i in gpu_indices_to_use:
        device = torch.device(f"cuda:{i}")
        
        # 根据配置的 MODE 决定目标显存大小
        if MODE == 'size':
            target_mem_bytes = int(TARGET_SIZE_GB * 1024**3)
            print(f"[cuda:{i}] 模式: 按指定大小。目标: {TARGET_SIZE_GB:.2f} GB")
        elif MODE == 'ratio':
            total_mem_on_device = torch.cuda.get_device_properties(device).total_memory
            target_mem_bytes = int(total_mem_on_device * TARGET_RATIO)
            print(f"[cuda:{i}] 模式: 按比例。目标: {TARGET_RATIO * 100:.1f}%")
        else:
            print(f"错误: 无效的MODE配置 '{MODE}'。请选择 'size' 或 'ratio'。")
            return

        thread = threading.Thread(target=allocate_and_run_on_device, args=(device, all_tensors, target_mem_bytes))
        threads.append(thread)
        thread.start()
        
    print("-" * 40)
    print("\n所有选定的GPU显存占用任务已启动。程序将保持运行。")
    print("要退出程序并释放显存，请按 Ctrl+C。")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n用户手动退出，正在停止程序。")
        # 正常情况下，主进程退出，守护线程也会退出，显存会自动释放
        # 这里不需要手动做什么
        os._exit(0) # 强制退出所有线程

if __name__ == "__main__":
    main()