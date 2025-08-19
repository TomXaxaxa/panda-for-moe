import json
import logging
import os
from functools import partial

import hydra
import numpy as np
import torch
import transformers
from panda.patchtst.dataset import TimeSeriesDataset
from panda.patchtst.evaluation import (
    evaluate_forecasting_model,
    evaluate_mlm_model,
)
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils import (
    get_dim_from_dataset,
    get_eval_data_dict,
    log_on_main,
    process_trajs,
    save_evaluation_results,
)

import torch.distributed as dist
from panda.utils import is_main_process
from panda.patchtst.deepseek_moe import MoEGate, DeepseekMLP

from collections import defaultdict

# from panda.patchtst.patchtst import TokenClusterGating

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)

ffn_inputs = defaultdict(list)

def get_ffn_input_hook(layer_idx):
    """一个工厂函数，创建并返回一个特定于层索引的钩子函数"""
    def hook(module, input, output):
        # input是一个元组，我们只需要第一个元素
        # 我们将其从计算图中分离，并移动到CPU以节省显存
        ffn_inputs[layer_idx].append(input[0].detach().cpu())
    return hook

# def log_moe_stats(model: torch.nn.Module):
#     """
#     聚合（支持DDP）并打印模型中所有MoE层的专家使用情况。
#     """
#     # 确保只在主进程执行打印和聚合后的最终计算
#     if not is_main_process():
#         # 在非主进程上，参与all_reduce但之后直接返回
#         if dist.is_initialized():
#             for module in model.modules():
#                 if isinstance(module, TokenClusterGating): # <-- 改动 1
#                     dist.all_reduce(module.expert_usage_count, op=dist.ReduceOp.SUM)
#                     dist.all_reduce(module.total_tokens_routed, op=dist.ReduceOp.SUM)
#         return
# 
#     # --- 以下代码仅在主进程上运行 ---
#     print("\n" + "="*50)
#     print(" " * 15 + "MoE Expert Usage Report")
#     print("="*50)
#     
#     found_moe_layer = False
#     # 遍历模型找到MoE门控模块
#     for i, module in enumerate(model.modules()):
#         if isinstance(module, TokenClusterGating): # <-- 改动 1
#             found_moe_layer = True
#             # 如果是分布式环境，需要从所有GPU聚合统计数据
#             if dist.is_initialized():
#                 # 主进程也参与all_reduce
#                 dist.all_reduce(module.expert_usage_count, op=dist.ReduceOp.SUM)
#                 dist.all_reduce(module.total_tokens_routed, op=dist.ReduceOp.SUM)
# 
#             if module.total_tokens_routed.item() == 0:
#                 print(f"MoE Gating Layer (approx. module {i}): No tokens were routed during evaluation. Skipping report.")
#                 continue
#                 
#             usage_percentages = module.expert_usage_count.float() / module.total_tokens_routed.item() * 100
#             
#             print(f"\n--- MoE Gating Layer (approx. module {i}) ---")
#             print(f"Total routing events during evaluation: {module.total_tokens_routed.item()}")
#             
#             # 使用正确的属性来获取专家数量
#             for exp_idx in range(module.num_experts): # <-- 改动 2
#                 print(
#                     f"  Expert {exp_idx:02d}: "
#                     f"routed {module.expert_usage_count[exp_idx].item():>8d} times "
#                     f"({usage_percentages[exp_idx]:.2f}%)"
#                 )
#     
#     if not found_moe_layer:
#         print("No MoE Gating layers (TokenClusterGating) found in the model.")
#         
#     print("="*50 + "\n")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    checkpoint_path = cfg.eval.checkpoint_path
    log(f"Using checkpoint: {checkpoint_path}")
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        log(f"Training info file found at: {training_info_path}")
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
            train_config = training_info.get("train_config", None)
            if train_config is None:  # for backwards compatibility
                train_config = training_info.get("training_config", None)
    else:
        log(f"No training info file found at: {training_info_path}")
        train_config = None

    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    pipeline = PatchTSTPipeline.from_pretrained(
        mode=cfg.eval.mode,
        pretrain_path=checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )
    model_config = dict(vars(pipeline.model.config))
    train_config = train_config or dict(cfg.train)
    # set floating point precision
    use_tf32 = train_config.get("tf32", False)
    log(f"use tf32: {use_tf32}")
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
        )
        use_tf32 = False

    rseed = train_config.get("seed", cfg.train.seed)
    log(f"Using SEED: {rseed}")
    transformers.set_seed(seed=rseed)

    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    channel_attention = model_config["channel_attention"]
    use_dynamics_embedding = model_config["use_dynamics_embedding"]

    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"eval prediction_length: {cfg.eval.prediction_length}")
    log(f"channel_attention: {channel_attention}")
    log(f"use_dynamics_embedding: {use_dynamics_embedding}")

    if channel_attention:
        # check use of channel rope
        channel_rope = model_config["channel_rope"]
        log(f"channel_rope: {channel_rope}")

    if use_dynamics_embedding:
        # check dynamics embedding parameters
        dynamics_embedding_config = {
            k: model_config[k]
            for k in [
                "num_poly_feats",
                "poly_degrees",
                "rff_trainable",
                "rff_scale",
                "num_rff",
            ]
        }
        log(f"dynamics embedding config: {dynamics_embedding_config}")

    # log("为每个Encoder层的FFN模块注册前向钩子以提取表征...")
    # encoder_layers = pipeline.model.model.encoder.layers
    # for i, layer in enumerate(encoder_layers):
    #     # 将钩子注册在ff模块上
    #     layer.ff.register_forward_hook(get_ffn_input_hook(i))
    #     log(f"  已为第 {i} 层注册钩子。")

    pipeline.model.eval()

    if hasattr(pipeline.model, "reset_moe_stats"):
        log("Resetting MoE statistics...")
        pipeline.model.reset_moe_stats()

    # for convenience, get system dimensions, for saving as a column in the metrics csv
    system_dims = {
        system_name: get_dim_from_dataset(test_data_dict[system_name][0])
        for system_name in test_data_dict
    }
    n_system_samples = {
        system_name: len(test_data_dict[system_name]) for system_name in test_data_dict
    }

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: TimeSeriesDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.eval.mode,
            mode="test",
        )
        for system_name in test_data_dict
    }

    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
    )
    process_trajs_fn = partial(
        process_trajs,
        split_coords=cfg.eval.split_coords,
        overwrite=cfg.eval.overwrite,
        verbose=cfg.eval.verbose,
    )
    log(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    if cfg.eval.mode == "predict":
        parallel_sample_reduction_fn = {
            "mean": lambda x: np.mean(x, axis=0),
            "median": lambda x: np.median(x, axis=0),
        }.get(cfg.eval.parallel_sample_reduction, lambda x: x)

        predictions, contexts, labels, metrics = evaluate_forecasting_model(
            pipeline,
            test_datasets,
            batch_size=cfg.eval.batch_size,
            prediction_length=cfg.eval.prediction_length,
            metric_names=cfg.eval.metric_names,
            return_predictions=cfg.eval.save_predictions,
            return_contexts=cfg.eval.save_contexts,
            return_labels=cfg.eval.save_labels,
            parallel_sample_reduction_fn=parallel_sample_reduction_fn,
            redo_normalization=True,
            prediction_kwargs=dict(
                sliding_context=cfg.eval.sliding_context,
                limit_prediction_length=cfg.eval.limit_prediction_length,
                verbose=cfg.eval.verbose,
            ),
            eval_subintervals=[
                (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
            ],
        )
        save_eval_results_fn(metrics)

        if cfg.eval.save_predictions and cfg.eval.save_contexts:
            assert predictions is not None and contexts is not None
            process_trajs_fn(
                cfg.eval.forecast_save_dir,
                {  # concatenate contexts and predictions
                    system: np.concatenate(
                        [contexts[system], predictions[system]], axis=2
                    )
                    for system in predictions
                },
            )

        if cfg.eval.save_labels and cfg.eval.save_contexts:
            assert labels is not None and contexts is not None
            process_trajs_fn(
                cfg.eval.labels_save_dir,
                {  # concatenate contexts and labels
                    system: np.concatenate([contexts[system], labels[system]], axis=2)
                    for system in labels
                },
            )

    elif cfg.eval.mode == "pretrain":
        completions, processed_past_values, timestep_masks, metrics = (
            evaluate_mlm_model(
                pipeline,
                test_datasets,
                metric_names=cfg.eval.metric_names,
                batch_size=cfg.eval.batch_size,
                undo_normalization=False,
                return_completions=cfg.eval.save_completions,
                return_processed_past_values=cfg.eval.save_contexts,
                return_masks=cfg.eval.save_masks,
            )
        )
        save_eval_results_fn(metrics)

        if cfg.eval.save_completions and completions is not None:
            process_trajs_fn(cfg.eval.completions_save_dir, completions)
        if cfg.eval.save_contexts and processed_past_values is not None:
            process_trajs_fn(cfg.eval.patch_input_save_dir, processed_past_values)
        if cfg.eval.save_masks and timestep_masks is not None:
            process_trajs_fn(cfg.eval.timestep_masks_save_dir, timestep_masks)
    else:
        raise ValueError(f"Invalid eval mode: {cfg.eval.mode}")
    
    # log("处理并保存收集到的FFN输入表征...")
    # final_representations = {}
    # for layer_idx, tensor_list in ffn_inputs.items():
    #     if not tensor_list:
    #         log(f"警告：第 {layer_idx} 层未收集到任何表征。")
    #         continue
    #     
    #     # 将所有批次的张量拼接成一个大张量
    #     log(f"  正在拼接第 {layer_idx} 层的 {len(tensor_list)} 个张量批次...")
    #     # PatchTST的FFN输入是3D的: (batch*channels, patches, d_model)
    #     # 我们需要将前两个维度合并以进行聚类
    #     concatenated_tensor = torch.cat(tensor_list, dim=0).view(-1, pipeline.model.config.d_model)
    #     final_representations[layer_idx] = concatenated_tensor
    #     log(f"  第 {layer_idx} 层最终表征形状: {concatenated_tensor.shape}")
# 
    # # 保存到文件
    # output_path = "ffn_representations.pt"
    # torch.save(final_representations, output_path)
    # log(f"所有层的令牌表征已成功保存到: {output_path}")
    # ======================== 新代码结束 ========================

#     log("Logging MoE statistics...")
#     log_moe_stats(pipeline.model)

if __name__ == "__main__":
    main()
