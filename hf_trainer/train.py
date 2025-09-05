#!/usr/bin/env python
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from os import getenv
from transformers import (
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from utils import (
    get_args, 
    get, 
    print_args, 
    get_seed, 
    extract_seed_from_ckpt,
    setup_logger,
    calculate_warmup_params,
    get_world_size
)
from models import create_model
from hf_trainer.dataset import create_dataset
from metrics import compute_metrics_factory
from trainer import VisionLanguageTrainer
from callbacks import SimplifiedProgressCallback, EpochCheckpointCallback


def main():
    # Parse config
    config, override = get_args()
    
    # Get configs
    train_config = get(config, 'trainor')
    val_config = get(config, 'validator')
    
    # Setup seed
    base_seed = get_seed()
    set_seed(base_seed)
    
    # Create checkpoint dir
    config.ckpt_dir = os.path.join(config.ckpt_dir, config.name)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # Determine if this is evaluation mode
    is_eval_only = (train_config.get('only_eval', False) or 
                   train_config.get('eval_only', False))
    
    # Determine main process and setup logger
    is_main = int(getenv("RANK", "0")) == 0


    # Extract seed from checkpoint if resuming
    if hasattr(config, 'ckpt') and config.get('ckpt') and not is_eval_only:
        config.ckpt = os.path.join(config.ckpt_dir, config.ckpt) if not os.path.exists(config.ckpt) else config.ckpt
        assert os.path.exists(config.ckpt), f"Path '{config.ckpt}' does not exist"
        seed = extract_seed_from_ckpt(config.ckpt)
    else:
        seed = base_seed

    logger = setup_logger(config.ckpt_dir, seed, is_main)

    logger.info("[Mode] Running in {} mode".format("EVALUATION" if is_eval_only else "TRAINING"))

    # Log if resuming from checkpoint
    if hasattr(config, 'ckpt') and config.get('ckpt') and not is_eval_only:
        logger.info(f"[Setup] Resuming training from checkpoint with seed: {seed}")
    
    # Log initialization
    logger.info("=" * 70)
    logger.info(f"[Setup] Initializing training with configuration:")
    logger.info(f"[Setup] Name: {config.name}")
    logger.info(f"[Setup] Seed: {seed}")
    logger.info(f"[Setup] Checkpoint directory: {config.ckpt_dir}")
    logger.info(f"[Setup] Main process: {is_main}")
    logger.info("=" * 70)
    
    # Dump config and print args (main process only)
    if is_main:
        config_path = f"{config.ckpt_dir}/config_{seed}_hf_trainer.json"
        json.dump(OmegaConf.to_container(config, resolve=True), 
                  open(config_path, "w"),
                  indent=4)
        logger.info(f"[Setup] Configuration saved to: {config_path}")
        # print_args(config, ['trainor', 'validator'], seed, override)
    

    # Create datasets
    logger.info("[Dataset] Loading datasets...")

    if is_eval_only:
        # Get evaluation splits from validator config
        eval_splits = val_config.get('splits', ['val'])
        assert isinstance(eval_splits, ListConfig), "splits must be a ListConfig"
        assert "train" not in eval_splits, "train split is not allowed in evaluation-only mode"
        assert hasattr(config, 'ckpt') and config.ckpt and config.ckpt is not None, "ckpt must be provided in evaluation-only mode"
        # TODO: right now, dataset crucial parameters doesnt rely on the content of the train split
        # TODO: max_len is an hyper parameters, vocab.tgt is saved during training
        # TODO: visual transform are set accordingly if split != "train"
        
        first_split = eval_splits[0]
        logger.info(f"[Dataset] Using '{first_split}' for dataset initialization")
        init_dataset = create_dataset(
            config,
            split=first_split,
            logger=logger
        )
        train_dataset = init_dataset
        val_dataset = init_dataset

    else:
        # Normal training mode - create both train and validation datasets
        train_dataset = create_dataset(
            config, 
            split='train',
            logger=logger
        )
        
        val_split = val_config.get('splits', ['val'])[0]
        val_dataset = create_dataset(
            config,
            split=val_split,
            logger=logger
        )
    
    # Create model
    logger.info("[Model] Creating model...")
    model = create_model(
        config=config,
        train_dataset=train_dataset,
        logger=logger
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Model] Total parameters: {num_params:,}")
    logger.info(f"[Model] Trainable parameters: {num_trainable:,}")
    
    # Calculate training parameters and warmup (only for training mode)
    if not is_eval_only:
        world_size = get_world_size()
        dataset_len = len(train_dataset)
        
        effective_batch_per_step, steps_per_epoch, total_training_steps, warmup_steps, warmup_ratio_pct = \
            calculate_warmup_params(train_config, dataset_len, world_size)
    else:
        # Set dummy values for eval-only mode
        world_size = get_world_size()
        dataset_len = len(val_dataset)
        effective_batch_per_step = val_config.batch_size
        steps_per_epoch = 1
        total_training_steps = 1
        warmup_steps = 0
        warmup_ratio_pct = 0
    
    # Log configuration
    logger.info("=" * 70)
    if is_eval_only:
        logger.info("[Evaluation Configuration]")
        logger.info(f"  Evaluation splits: {eval_splits}")
        logger.info(f"  World size (GPUs): {world_size}")
        logger.info(f"  Batch size: {val_config.batch_size}")
        logger.info(f"  Beam width: {val_config.get('beam_width', 2)}")
        logger.info(f"  Max generation length: {val_config.get('gen_max_length', 128)}")
        logger.info(f"  Metrics: {val_config.get('metrics', ['radevalbertscore'])}")
    else:
        logger.info("[Training Configuration]")
        logger.info(f"  Dataset size: {dataset_len:,}")
        logger.info(f"  World size (GPUs): {world_size}")
        logger.info(f"  Per-device batch size: {train_config.batch_size}")
        logger.info(f"  Gradient accumulation steps: {train_config.get('grad_accu', 1)}")
        logger.info(f"  Effective batch size per step: {effective_batch_per_step}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total training steps: {total_training_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps:,} ({warmup_ratio_pct}% of total)")
        logger.info(f"  Learning rate: {train_config.optim_params.lr:.2e}")
        logger.info(f"  Weight decay: {train_config.optim_params.get('weight_decay', 0.0)}")
        logger.info(f"  Scheduler: Linear warmup + Cosine decay")
        logger.info(f"  Mixed precision (AMP): {train_config.get('use_amp', False)}")
    logger.info("=" * 70)
    
    # Configure training arguments with proper epoch-based checkpointing
    training_args = TrainingArguments(
        output_dir=config.ckpt_dir,
        overwrite_output_dir=False,
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=val_config.batch_size,
        gradient_accumulation_steps=train_config.get('grad_accu', 1),
        gradient_checkpointing=False,
        eval_strategy="epoch" if not is_eval_only else "no",
        eval_delay=train_config.get('eval_start', 0),
        save_strategy="epoch" if not is_eval_only else "no",
        save_total_limit=None,  # Keep all checkpoints
        learning_rate=train_config.optim_params.lr,
        weight_decay=train_config.optim_params.get('weight_decay', 0.0),
        adam_beta1=train_config.optim_params.get('betas', [0.9, 0.999])[0],
        adam_beta2=train_config.optim_params.get('betas', [0.9, 0.999])[1],
        adam_epsilon=train_config.optim_params.get('eps', 1e-8),
        max_grad_norm=train_config.get('clip_grad_norm', 1.0),
        warmup_steps=warmup_steps if not is_eval_only else 0,
        lr_scheduler_type="cosine" if not is_eval_only else "constant",
        logging_dir=os.path.join(config.ckpt_dir, 'logs'),
        logging_strategy="steps" if not is_eval_only else "no",
        logging_steps=50,
        metric_for_best_model=train_config.get('early_stop_metric', 'radevalbertscore'),
        greater_is_better=True,
        load_best_model_at_end=True if not is_eval_only else False,
        save_safetensors=False,
        fp16=train_config.get('use_amp', False),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False,
        log_level="error",  # Suppress default verbose logging
        log_level_replica="error",
        log_on_each_node=False,
        disable_tqdm=True,  # Disable tqdm since we have custom progress logging
    )
    
    # Create compute_metrics function with seed
    logger.info(f"[Metrics] Setting up evaluation metrics: {val_config.get('metrics', ['radevalbertscore'])}")
    metrics_context_split = None
    if not is_eval_only:
        # In training mode, tag predictions with the validation split name
        metrics_context_split = val_config.get('splits', ['val'])[0]
    compute_metrics_fn = compute_metrics_factory(
        val_config.get('metrics', ['radevalbertscore']),
        train_dataset.tokenizer,
        save_dir=config.ckpt_dir,
        logger=logger,
        seed=seed,
        is_main_process=is_main,
        context={'split': metrics_context_split} if metrics_context_split else None
    )
    
    # Setup callbacks
    callbacks = [
        SimplifiedProgressCallback(logger=logger),  # Custom simplified logging
        EpochCheckpointCallback(config.ckpt_dir, seed=seed, save_epochs=True, logger=logger),  # Checkpoint with seed
    ]
    
    # Add early stopping if configured and not in eval mode
    if not is_eval_only:
        early_stop_patience = train_config.get('early_stop', None)
        if early_stop_patience:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stop_patience,
                    early_stopping_threshold=0.0
                )
            )
            logger.info(f"[Training] Early stopping enabled with patience={early_stop_patience}")
    
    # Create trainer
    trainer = VisionLanguageTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.get_collate_fn(),
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    
    # Set beam width and processing class for generation/decoding
    trainer.beam_width = val_config.get('beam_width', 2)
    trainer.processing_class = train_dataset.tokenizer
    trainer.gen_max_length = val_config.get('gen_max_length', 128)
    
    logger.info(f"[Generation] Beam width: {trainer.beam_width}")
    logger.info(f"[Generation] Max generation length: {trainer.gen_max_length}")
    
    
    # Evaluation-only mode
    if is_eval_only:
        logger.info("=" * 70)
        logger.info(f"[Evaluation] Starting evaluation on splits: {eval_splits}")
        logger.info(f"[Evaluation] Using checkpoint: {config.ckpt}")

        for split_name in eval_splits:
            logger.info("-" * 70)
            logger.info(f"[Evaluation] Split: '{split_name}' - creating dataset")
            eval_dataset = create_dataset(
                config,
                split=split_name,
                logger=logger
            )

            # Update compute_metrics with split context and (optionally) tokenizer
            trainer.compute_metrics = compute_metrics_factory(
                val_config.get('metrics', ['radevalbertscore']),
                eval_dataset.tokenizer if hasattr(eval_dataset, 'tokenizer') else train_dataset.tokenizer,
                save_dir=config.ckpt_dir,
                logger=logger,
                seed=seed,
                is_main_process=is_main,
                context={'split': split_name}
            )

            trainer._load_from_checkpoint(config.ckpt)
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            
            # Log metrics
            logger.info(f"[Evaluation] Results for '{split_name}' split:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

            # Save evaluation results with split name in filename
            if is_main:
                eval_results_path = os.path.join(config.ckpt_dir, f"eval_results_{split_name}_seed{seed}.json")
                with open(eval_results_path, "w") as f:
                    json.dump({
                        'split': split_name,
                        'seed': seed,
                        'checkpoint': config.ckpt,
                        'metrics': metrics
                    }, f, indent=4)
                logger.info(f"[Evaluation] Results saved to: {eval_results_path}")

        logger.info("=" * 70)
        logger.info("[Evaluation] All evaluations completed successfully!")
        logger.info("=" * 70)
        return

    # Training mode
    logger.info("=" * 70)
    logger.info("[Training] Starting training...")
    
    # Handle checkpoint resuming
    resume_checkpoint = None
    if hasattr(config, 'ckpt') and config.ckpt:
        resume_checkpoint = config.ckpt
        logger.info(f"[Training] Resuming from checkpoint: {resume_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    final_model_path = os.path.join(config.ckpt_dir, f"final_model_seed{seed}")
    trainer.save_model(final_model_path)
    logger.info(f"[Training] Final model saved to: {final_model_path}")
    
    logger.info("=" * 70)
    logger.info("[Training] Training completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()