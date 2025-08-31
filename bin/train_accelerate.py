#!/usr/bin/env python3
"""
Accelerate-optimized training script for vilmedic.
This provides significantly better multi-GPU performance than DataParallel.

Usage:
    # Single GPU
    python bin/train_accelerate.py --config configs/your_config.yaml

    # Multi-GPU with accelerate (recommended)
    accelerate launch bin/train_accelerate.py --config configs/your_config.yaml
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=gpu bin/train_accelerate.py --config configs/your_config.yaml
    
    # Specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 bin/train_accelerate.py --config configs/your_config.yaml
"""

import os
import sys
import argparse
import logging
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vilmedic.executors.trainor_accelerate import AcceleratedTrainor
from vilmedic.executors.validator import Validator


def setup_logging(seed, is_main_process=True):
    """Setup logging configuration."""
    logger = logging.getLogger(str(seed))
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    
    if is_main_process:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        # file_handler = logging.FileHandler(f'train_{seed}.log')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
    
    # Add a custom method for settings logging
    def settings(self, msg):
        self.info(f"[SETTINGS] {msg}")
    
    logger.settings = lambda msg: settings(logger, msg)
    
    return logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Accelerate-optimized training for vilmedic')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validate_every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--no_validation', action='store_true', help='Disable validation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Ensure checkpoint directory exists
    if hasattr(config, 'ckpt_dir'):
        os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # Create trainor with Accelerate
    trainor = AcceleratedTrainor(config=config, seed=args.seed)
    
    # Setup logging (only on main process)
    logger = setup_logging(args.seed, trainor.accelerator.is_main_process)
    
    # Log configuration
    if trainor.accelerator.is_main_process:
        logger.info("="*70)
        logger.info("ACCELERATE-OPTIMIZED TRAINING")
        logger.info("="*70)
        logger.info(f"Config: {args.config}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Device: {trainor.accelerator.device}")
        logger.info(f"Num processes: {trainor.accelerator.num_processes}")
        logger.info(f"Mixed precision: {trainor.accelerator.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {trainor.grad_accu}")
        
        if trainor.accelerator.distributed_type.name != "NO":
            logger.info(f"Distributed training: {trainor.accelerator.distributed_type.name}")
            logger.info(f"Process index: {trainor.accelerator.process_index}")
        
        # Performance tips
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE OPTIMIZATIONS ACTIVE:")
        logger.info("  ✓ Accelerate DDP (instead of DataParallel)")
        logger.info("  ✓ Optimized DataLoader (persistent_workers, prefetch)")
        if trainor.accelerator.mixed_precision != 'no':
            logger.info(f"  ✓ Mixed precision training ({trainor.accelerator.mixed_precision})")
        if trainor.grad_accu > 1:
            logger.info(f"  ✓ Gradient accumulation ({trainor.grad_accu} steps)")
        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
            logger.info("  ✓ Gradient checkpointing")
        logger.info("="*70 + "\n")
    
    # Create validator if needed
    if not args.no_validation:
        # Only create validator on main process
        if trainor.accelerator.is_main_process:
            validator_config = config.copy()
            validator_config.batch_size = config.get('val_batch_size', config.batch_size)
            
            # Create validator with unwrapped model
            validator = Validator(
                config=validator_config,
                seed=args.seed,
                ckpt=trainor.ckpt,
                split='val'
            )
            
            # Set the model (unwrapped version)
            validator.models = [trainor.accelerator.unwrap_model(trainor.model)]
            trainor.evaluator = validator
            
            logger.info("Validator created for evaluation")
    
    # Start training
    try:
        trainor.start()
    except KeyboardInterrupt:
        if trainor.accelerator.is_main_process:
            logger.info("\nTraining interrupted by user")
    except Exception as e:
        if trainor.accelerator.is_main_process:
            logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up
        trainor.accelerator.end_training()
        if trainor.accelerator.is_main_process:
            logger.info("Training session ended")


if __name__ == "__main__":
    main()
