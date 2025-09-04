import logging
import os
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from omegaconf import OmegaConf
from panda.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
    RandomDimSelectionTransform,
    RandomFourierSeries,
    RandomPhaseSurrogate,
    RandomTakensEmbedding,
    StandardizeTransform,
)
from panda.patchtst.dataset import TimeSeriesDataset
# 确保导入了 PatchTSTForPrediction 以便在第二阶段加载模型
from panda.patchtst.patchtst import (
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)
from panda.schedulers import Scheduler, SchedulerLoggingCallback
from panda.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    load_patchtst_model,
    log_on_main,
    save_training_info,
)
from transformers import (
    Trainer,
    TrainingArguments,
)

import wandb

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: PatchTSTForPretraining | PatchTSTForPrediction,
        args: TrainingArguments,
        scheduler: Scheduler,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.scheduler = scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        epoch = float(self.state.epoch)  # type: ignore
        schedule_param = torch.tensor(self.scheduler(epoch))

        outputs = model(**inputs, schedule_param=schedule_param)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later (HF comment)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # =========================================================================
    # == 1. 初始化设置 (W&B, 日志, 随机种子, 精度等)
    # =========================================================================
    if cfg.wandb.log and is_main_process():
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=dict(cfg),
            sync_tensorboard=False,
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
            id=cfg.wandb.resume_run_id,
        )
        log_on_main(f"Wandb initialized: {run.id}", logger)

    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    output_dir = get_next_path(
        cfg.run_name if cfg.run_name else "run",
        base_dir=Path(cfg.train.output_dir),
        file_type="",
        overwrite=cfg.train.resume_from_checkpoint is not None,
    )
    log_on_main(f"Logging dir: {output_dir}", logger)
    
    # =========================================================================
    # == 2. 数据集准备 (加载, 过滤, 增强)
    # =========================================================================
    train_data_dir_lst = cfg.train_data_dirs
    train_data_paths = []
    for train_data_dir in train_data_dir_lst:
        train_data_dir = os.path.expandvars(train_data_dir)
        train_data_paths.extend(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
        )
    
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets for training from directories: {train_data_dir_lst}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.patchtst.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h", one_dim_target=False),
        )
        for data_path in train_data_paths
    ]

    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)
    assert len(train_datasets) == len(probability)

    dataloader_num_workers = cfg.train.dataloader_num_workers
    if dataloader_num_workers > len(train_datasets):
        log_on_main(
            f"Setting the number of data loader workers to {len(train_datasets)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(train_datasets)

    augmentations = [
        RandomTakensEmbedding(
            lag_range=cfg.augmentations.lag_range,
            random_seed=cfg.train.seed,
        ),
        RandomConvexCombinationTransform(
            alpha=1.0,
            random_seed=cfg.train.seed,
            dim_range=cfg.augmentations.dim_range,
        ),
        RandomAffineTransform(
            dim_range=cfg.augmentations.dim_range,
            scale=1.0,
            random_seed=cfg.train.seed,
        ),
        RandomPhaseSurrogate(
            cutoff=cfg.augmentations.phase_surrogate_cutoff,
            random_seed=cfg.train.seed,
        ),
        RandomFourierSeries(
            max_wavenumber=cfg.augmentations.max_wavenumber,
            max_amp=cfg.augmentations.max_amp,
            mode_range=cfg.augmentations.mode_range,
            random_seed=cfg.train.seed,
        ),
    ]
    if cfg.augmentations.probabilities is None:
        cfg.augmentations.probabilities = [1.0 / len(augmentations)] * len(augmentations)
    else:
        cfg.augmentations.probabilities = [
            prob / sum(cfg.augmentations.probabilities)
            for prob in cfg.augmentations.probabilities
        ]

    transforms: list = [
        StandardizeTransform(),
        RandomDimSelectionTransform(num_dims=cfg.fixed_dim),
    ]

    # 这个数据集将被两个训练阶段共享
    shuffled_train_dataset = TimeSeriesDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="train",
        model_type=cfg.patchtst.mode,
        augmentations=augmentations,
        augmentation_probabilities=cfg.augmentations.probabilities,
        augmentation_rate=cfg.augmentations.augmentation_rate,
        transforms=transforms,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)


    # =========================================================================
    # == STAGE 1: 预训练基础 Transformer 模型
    # =========================================================================
    log_on_main("="*80, logger)
    log_on_main("== 开始第一阶段: 预训练基础模型 ==", logger)
    log_on_main("="*80, logger)

    # 确保在第一阶段禁用 prompt network
    cfg.patchtst.use_prompt_network = False
    
    log_on_main("Initializing model for Stage 1", logger)
    model_stage1 = load_patchtst_model(
        mode=cfg.patchtst.mode,
        model_config=dict(cfg.patchtst),
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
        pretained_checkpoint=cfg.patchtst.pretrained_pft_path,
    )

    trainable_params_s1 = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
    log_on_main(f"第一阶段可训练参数量: {trainable_params_s1:,}", logger)

    output_dir_s1 = Path(output_dir) / "stage1"
    training_args_s1 = TrainingArguments(
        run_name=f"{cfg.run_name}-S1",
        output_dir=str(output_dir_s1),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        log_on_each_node=False,
        logging_dir=str(output_dir_s1 / "logs"),
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["wandb"] if cfg.wandb.log else ["tensorboard"],
        max_steps=cfg.train.max_steps, # 使用配置中的max_steps
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        tf32=use_tf32,
        bf16=True,
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        ddp_backend=cfg.train.ddp_backend,
        remove_unused_columns=cfg.train.remove_unused_columns,
        seed=cfg.train.seed,
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
        push_to_hub=False,
    )

    ensure_contiguous(model_stage1)

    # 根据配置设置训练器 (带或不带自定义调度器)
    scheduler_args = dict(cfg.scheduler)
    if scheduler_args.pop("enabled", False):
        scheduler = Scheduler(**scheduler_args)
        logging_callback = SchedulerLoggingCallback(scheduler=scheduler, logger=logger, log_interval=cfg.train.log_steps)
        trainer_stage1 = CustomTrainer(
            model=model_stage1, args=training_args_s1, train_dataset=shuffled_train_dataset,
            scheduler=scheduler, callbacks=[logging_callback]
        )
    else:
        trainer_stage1 = Trainer(
            model=model_stage1, args=training_args_s1, train_dataset=shuffled_train_dataset
        )
    
    log_on_main("开始第一阶段训练...", logger)
    trainer_stage1.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)

    stage1_final_checkpoint = Path(output_dir_s1) / "checkpoint-final"
    if is_main_process():
        model_stage1.save_pretrained(stage1_final_checkpoint)
        log_on_main(f"第一阶段模型已保存至: {stage1_final_checkpoint}", logger)
        
    torch.distributed.barrier()


    # =========================================================================
    # == STAGE 2: 微调 Prompt Network
    # =========================================================================
    log_on_main("\n" + "="*80, logger)
    log_on_main("== 开始第二阶段: 微调 Prompt Network ==", logger)
    log_on_main("="*80, logger)

    # 为第二阶段启用 prompt network
    cfg.patchtst.use_prompt_network = True
    
    # 必须从第一阶段的checkpoint加载模型
    # `strict=False` 很重要, 因为 prompt network 的权重是新初始化的, 不在 checkpoint 中
    log_on_main(f"从 {stage1_final_checkpoint} 加载模型用于第二阶段...", logger)
    # 更新config以确保模型在加载时知道要构建prompt network
    model_stage1.config.use_prompt_network = True
    model_stage2 = PatchTSTForPrediction.from_pretrained(
        stage1_final_checkpoint,
        config=model_stage1.config,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    
    log_on_main("为第二阶段冻结 Transformer 层...", logger)
    for name, param in model_stage2.named_parameters():
        param.requires_grad = False # 首先冻结所有参数
        
        # 然后解冻需要的层
        if "prompt_network" in name or "fft_projector" in name or "head.projection" in name:
            param.requires_grad = True
        if "norm" in name.lower():
            param.requires_grad = True
        if "embedder" in name or "patchifier" in name or "scaler" in name:
            param.requires_grad = True
            
    trainable_params_s2 = sum(p.numel() for p in model_stage2.parameters() if p.requires_grad)
    log_on_main(f"第二阶段可训练参数量: {trainable_params_s2:,}", logger)

    output_dir_s2 = Path(output_dir) / "stage2"
    # 为第二阶段微调设置独立的训练参数 (更小的学习率, 不同的步数)
    training_args_s2 = TrainingArguments(
        run_name=f"{cfg.run_name}-S2-Prompt",
        output_dir=str(output_dir_s2),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        # 建议为第二阶段在config中设置独立的学习率和步数
        learning_rate=cfg.train.learning_rate / 2500,
        max_steps=cfg.train.max_steps,
        # 复制其他通用参数
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        log_on_each_node=False,
        logging_dir=str(output_dir_s2 / "logs"),
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["wandb"] if cfg.wandb.log else ["tensorboard"],
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        tf32=use_tf32,
        bf16=True,
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        ddp_backend=cfg.train.ddp_backend,
        remove_unused_columns=cfg.train.remove_unused_columns,
        seed=cfg.train.seed,
        push_to_hub=False,
    )

    ensure_contiguous(model_stage2)

    # 第二阶段通常不需要复杂的学习率调度器, 但我们保持结构一致
    if scheduler_args.get("enabled", False):
        scheduler_s2 = Scheduler(**scheduler_args)
        logging_callback_s2 = SchedulerLoggingCallback(scheduler=scheduler_s2, logger=logger, log_interval=cfg.train.log_steps)
        trainer_stage2 = CustomTrainer(
            model=model_stage2, args=training_args_s2, train_dataset=shuffled_train_dataset,
            scheduler=scheduler_s2, callbacks=[logging_callback_s2]
        )
    else:
        trainer_stage2 = Trainer(
            model=model_stage2, args=training_args_s2, train_dataset=shuffled_train_dataset
        )

    log_on_main("开始第二阶段训练...", logger)
    trainer_stage2.train() # 第二阶段从头开始训练, 不恢复

    if is_main_process():
        final_model_dir = Path(output_dir) / "checkpoint-final"
        model_stage2.save_pretrained(final_model_dir)
        save_training_info(
            final_model_dir,
            model_config=OmegaConf.to_container(cfg.patchtst, resolve=True),
            train_config=OmegaConf.to_container(cfg.train, resolve=True),
            all_config=OmegaConf.to_container(cfg, resolve=True),
        )
        log_on_main(f"最终微调后的模型已保存至: {final_model_dir}", logger)

    if cfg.wandb.log:
        wandb.finish(exit_code=0)


if __name__ == "__main__":
    main()