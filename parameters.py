from transformers import AutoModel

# --- 请将这里替换成您的模型文件夹路径 ---
model_path = "./checkpoints/panda256+encoder+Moirai-MOE/checkpoint-final" 

try:
    # 从本地路径加载模型
    # 这会自动读取 config.json 和 model.safetensors
    model = AutoModel.from_pretrained(model_path)

    # 计算总参数量
    total_params = model.num_parameters()
    
    # 为了可读性，计算可训练的参数量（在这种情况下应该与总参数量相同）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型路径: {model_path}")
    print("------------------------------------------")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("------------------------------------------")

    # 以更易读的单位（百万 M / 十亿 B）显示
    if total_params >= 1_000_000_000:
        print(f"总参数量 (约): {total_params / 1_000_000_000:.2f} B")
    else:
        print(f"总参数量 (约): {total_params / 1_000_000:.2f} M")

except Exception as e:
    print(f"加载模型时出错: {e}")
    print("请确保路径正确，并且模型文件没有损坏。")