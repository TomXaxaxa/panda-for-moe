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

from panda.patchtst.modules import PatchTSTRMSNorm

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


def setup_model_for_stage2_training(model):
    """
    Freezes the main transformer blocks and keeps only the specified layers trainable.
    """
    log_on_main("--- Setting up model for Stage 2 training ---", logger)
    
    # First, freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Specifically unfreeze the parameters of the prompt network and other specified layers
    # Note: Accessing nested modules might require careful path inspection
    # Based on our previous changes to PatchTSTModel
    if hasattr(model.model, 'fft_proj'):
        for param in model.model.fft_proj.parameters():
            param.requires_grad = True
        log_on_main("Unfroze fft_proj parameters.", logger)

    if hasattr(model.model, 'freq_memory_prompt'):
        for param in model.model.freq_memory_prompt.parameters():
            param.requires_grad = True
        log_on_main("Unfroze freq_memory_prompt parameters.", logger)


    # Unfreeze embedding layer
    if hasattr(model.model.encoder, 'embedder'):
        for param in model.model.encoder.embedder.parameters():
            param.requires_grad = True
        log_on_main("Unfroze encoder.embedder parameters.", logger)
            
    # Unfreeze all normalization layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, PatchTSTRMSNorm)):
            for param in module.parameters():
                param.requires_grad = True
    log_on_main("Unfroze all LayerNorm and PatchTSTRMSNorm parameters.", logger)


    # Log the number of trainable parameters to confirm our setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_on_main(f"Stage 2: Trainable parameters: {trainable_params:,} (out of {total_params:,})", logger)
    
    return model


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set up wandb project and logging if enabled
    if cfg.wandb.log and is_main_process():
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=dict(cfg),
            sync_tensorboard=False,  # auto-upload tensorboard metrics
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
            id=cfg.wandb.resume_run_id,
        )
        log_on_main(f"Wandb initialized: {run.id}", logger)

    # set floating point precision
    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    # get train data paths
    train_data_dir_lst = cfg.train_data_dirs
    train_data_paths = []
    for train_data_dir in train_data_dir_lst:
        train_data_dir = os.path.expandvars(train_data_dir)
        train_data_paths.extend(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
        )
    # create a new output directory to save results
    output_dir = get_next_path(
        cfg.run_name if cfg.run_name else "run",
        base_dir=Path(cfg.train.output_dir),
        file_type="",
        overwrite=cfg.train.resume_from_checkpoint is not None,
    )

    log_on_main(f"Logging dir: {output_dir}", logger)
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
            FileDataset(path=Path(data_path), freq="h", one_dim_target=False),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    # set probabilities (how we weight draws from each data file)
    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)
    assert len(train_datasets) == len(probability)

    # adapt number of workers to the number of datasets if there are more workers than datasets
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
        cfg.augmentations.probabilities = [1.0 / len(augmentations)] * len(
            augmentations
        )
    else:  # ensure probabilities sum to 1
        cfg.augmentations.probabilities = [
            prob / sum(cfg.augmentations.probabilities)
            for prob in cfg.augmentations.probabilities
        ]

    log_on_main(
        f"Using augmentations: {[aug for aug, prob in zip(augmentations, cfg.augmentations.probabilities) if prob > 0.0]}",
        logger,
    )

    transforms: list = [
        StandardizeTransform(),
        RandomDimSelectionTransform(num_dims=cfg.fixed_dim),
    ]

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

    if (
        cfg.patchtst.mode == "predict"
        and cfg.patchtst.pretrained_encoder_path is not None
    ):
        log_on_main(
            f"Loading pretrained encoder from {cfg.patchtst.pretrained_encoder_path}",
            logger,
        )

    log_on_main("Initializing model", logger)

    model = load_patchtst_model(
        mode=cfg.patchtst.mode,
        model_config=dict(cfg.patchtst),
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
        pretained_checkpoint=cfg.patchtst.pretrained_pft_path,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main(f"Total trainable parameters: {trainable_params:,}", logger)

    # Define training args
    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        log_on_each_node=False,
        logging_dir=str(output_dir / "logs")
        if not (cfg.wandb.log and is_main_process())
        else f"wandb/{run.name}_{run.id}/logs",
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["wandb"] if cfg.wandb.log else ["tensorboard"],
        max_steps=cfg.train.max_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        tf32=use_tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        bf16=True,      # using only when flash attention is enabled
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        ddp_backend=cfg.train.ddp_backend,
        remove_unused_columns=cfg.train.remove_unused_columns,
        seed=cfg.train.seed,
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
    )

    # check if model weights are contiguous in memory; if not, make them contiguous tensors.
    # This speeds up training and allows checkpoint saving by transformers Trainer
    ensure_contiguous(model)

    scheduler_args = dict(cfg.scheduler)
    scheduler_enabled = scheduler_args.pop("enabled", False)
    
    if scheduler_enabled:
        log_on_main(
            f"Using {scheduler_args['schedule_name']} scheduler for {scheduler_args['schedule_value_name']}",
            logger,
        )
        value_name = scheduler_args.pop("schedule_value_name", "schedule_param")
        scheduler = Scheduler(**scheduler_args)

        logging_callback = SchedulerLoggingCallback(
            scheduler=scheduler,
            logger=logger,
            log_interval=cfg.train.log_steps,
            log_value_name=value_name,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_train_dataset,
            scheduler=scheduler,
            callbacks=[logging_callback],
        )
    else:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=shuffled_train_dataset
        )

    # === STAGE 1: FULL MODEL TRAINING ===
    log_on_main("--- Starting Stage 1: Training all model parameters ---", logger)
    
    # --- 用于从断点恢复的临时修改：开始 ---
    # 由于第一阶段已经完成，我们注释掉它的训练过程以直接进入第二阶段
    log_on_main("--- SKIPPING Stage 1 training to resume at Stage 2 ---", logger)
    # trainer.train(
    #     resume_from_checkpoint=cfg.train.resume_from_checkpoint
    # ) 
    # --- 用于从断点恢复的临时修改：结束 ---

    # 原始代码（已注释掉，供参考）:
    # trainer.train(
    #     resume_from_checkpoint=cfg.train.resume_from_checkpoint
    # )

    # 即使跳过了训练，我们仍然需要定义第一阶段检查点文件夹的路径
    stage1_final_checkpoint_dir = output_dir / "checkpoint-final-stage1"
    
    # 这部分保存代码在跳过训练时不会执行，是正常的
    if is_main_process() and trainer.state.is_world_process_zero and not os.path.exists(stage1_final_checkpoint_dir):
         log_on_main(f"Saving final Stage 1 model to {stage1_final_checkpoint_dir}", logger)
         # model.save_pretrained(stage1_final_checkpoint_dir) # 在跳过时，模型未训练，无需保存

    log_on_main("--- Stage 1 Training Finished (or Skipped) ---", logger)

    # === STAGE 2: FINE-TUNING PROMPT NETWORK & EMBEDDINGS ===
    log_on_main("--- Starting Stage 2: Fine-tuning specified layers ---", logger)

    # 1. Reload the model from Stage 1
    # --- 用于从断点恢复的临时修改：开始 ---
    # !!! 重要：请将下面的路径替换为你第一阶段实际的输出路径 !!!
    stage1_checkpoint_path = "./checkpoints/run-75/checkpoint-final-stage1" 
    log_on_main(f"Loading model from Stage 1 checkpoint: {stage1_checkpoint_path}", logger)
    model = load_patchtst_model(
        mode=cfg.patchtst.mode,
        model_config=dict(cfg.patchtst),
        pretained_checkpoint=stage1_checkpoint_path, # 使用你指定的路径
    )
    # --- 用于从断点恢复的临时修改：结束 ---

    # 原始代码（已注释掉，供参考）:
    # log_on_main(f"Loading model from Stage 1 checkpoint: {stage1_final_checkpoint_dir}", logger)
    # model = load_patchtst_model(
    #     mode=cfg.patchtst.mode,
    #     model_config=dict(cfg.patchtst),
    #     pretained_checkpoint=str(stage1_final_checkpoint_dir),
    # )
    
    ensure_contiguous(model)

    # 2. Freeze the transformer layers and set up for Stage 2
    model = setup_model_for_stage2_training(model)

    # 3. Create a new output directory and TrainingArguments for Stage 2
    stage2_output_dir = output_dir / "stage2_finetune"
    stage2_output_dir.mkdir(exist_ok=True, parents=True)
    
    stage2_training_args_dict = training_args.to_dict()
    stage2_training_args_dict['output_dir'] = str(stage2_output_dir)
    stage2_training_args_dict['resume_from_checkpoint'] = None
    
    stage2_training_args = TrainingArguments(**stage2_training_args_dict)

    # 4. Instantiate a new Trainer for Stage 2
    if scheduler_enabled:
        trainer_stage2 = CustomTrainer(
            model=model,
            args=stage2_training_args,
            train_dataset=shuffled_train_dataset,
            scheduler=scheduler,
            callbacks=[logging_callback],
        )
    else:
        trainer_stage2 = Trainer(
            model=model, args=stage2_training_args, train_dataset=shuffled_train_dataset
        )
    
    # 5. Run Stage 2 training
    log_on_main("Starting Stage 2 training loop", logger)
    trainer_stage2.train()

    # 6. Save the final fine-tuned model
    if is_main_process():
        final_stage2_dir = stage2_output_dir / "checkpoint-final"
        model.save_pretrained(final_stage2_dir)
        save_training_info(
            final_stage2_dir,
            model_config=OmegaConf.to_container(cfg.patchtst, resolve=True),
            train_config=OmegaConf.to_container(cfg.train, resolve=True),
            all_config=OmegaConf.to_container(cfg, resolve=True),
        )

    # terminate wandb run after training
    if cfg.wandb.log and is_main_process():
        wandb.finish(exit_code=0)


if __name__ == "__main__":
    main()