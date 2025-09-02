#!/usr/bin/env python
"""
Accelerate-enabled training script for vilmedic.
This script leverages Hugging Face Accelerate for distributed training across multiple GPUs.
"""

import json
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed, InitProcessGroupKwargs
from vilmedic.executors import Validator
from vilmedic.executors.validator_accelerate import ValidatorAccelerate
from utils import get_args, get, print_args, get_seed, extract_seed_from_ckpt
from logger import set_logger
from omegaconf import OmegaConf

# Import our custom accelerate-enabled trainor
from vilmedic.executors.trainor_accelerate import TrainorAccelerate


def main():
    # Parse config first to get gradient accumulation steps
    config, override = get_args()
    
    # Get gradient accumulation steps from config
    train_config = get(config, 'trainor')
    grad_accu_steps = train_config.get('grad_accu', 1) if hasattr(train_config, 'grad_accu') else 1
    use_amp = train_config.get('use_amp', False) if hasattr(train_config, 'use_amp') else False
    
    # Initialize accelerator with proper gradient accumulation and increased timeout for evaluation
    # Increase NCCL timeout to 30 minutes (1800 seconds) to handle long evaluation runs
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    
    accelerator = Accelerator(
        mixed_precision='fp16' if use_amp else 'no',
        gradient_accumulation_steps=grad_accu_steps,
        log_with=None,  # We use our own logging
        kwargs_handlers=[init_kwargs]  # Pass the timeout configuration
    )
    # Get base seed and modify for distributed training
    base_seed = get_seed()
    # Each process gets a different seed to ensure different random states
    seed = base_seed + accelerator.process_index
    accelerate_set_seed(seed)
    
    # Only main process handles directory creation and logging setup
    if accelerator.is_main_process:
        # Create checkpoint dir
        config.ckpt_dir = os.path.join(config.ckpt_dir, config.name)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        
        # If ckpt is specified, we continue training. Let's extract seed
        if hasattr(config, 'ckpt') and config.get('ckpt'):
            config.ckpt = os.path.join(config.ckpt_dir, config.ckpt) if not os.path.exists(config.ckpt) else config.ckpt
            assert os.path.exists(config.ckpt), f"Path '{config.ckpt}' does not exist"
            seed = extract_seed_from_ckpt(config.ckpt)
        
        # Create logger according to seed
        set_logger(config.ckpt_dir, seed)
        
        # Dump config
        json.dump(OmegaConf.to_container(config, resolve=True), 
                  open(f"{config.ckpt_dir}/config_{seed}_accelerate.json", "w"),
                  indent=4)
        
        # Nice print args
        print_args(config, ['trainor', 'validator'], seed, override)
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    
    # Fetch args for validation (train_config already fetched above)
    val_config = get(config, 'validator')
    
    # Create Accelerate-enabled Trainor
    trainor = TrainorAccelerate(
        config=train_config,
        seed=seed,
        accelerator=accelerator
    )
    
    # Create evaluator - use distributed version for multi-GPU
    if accelerator.num_processes > 1:
        # Multi-GPU: use ValidatorAccelerate for distributed evaluation
        evaluator = ValidatorAccelerate(
            config=val_config,
            model=trainor.model,  # Pass the wrapped model
            train_dl=trainor.dl,
            seed=seed,
            from_training=True,
            accelerator=accelerator
        )
    else:
        # Single GPU: use regular validator on main process
        evaluator = Validator(
            config=val_config,
            models=[trainor.model],
            train_dl=trainor.dl,
            seed=seed,
            from_training=True
        ) if accelerator.is_main_process else None
    
    trainor.evaluator = evaluator
    
    # Log distributed training info
    if accelerator.is_main_process:
        logger = logging.getLogger(str(seed))
        logger.info(f"Accelerate training initialized:")
        logger.info(f"  Number of processes: {accelerator.num_processes}")
        logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"  Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
        logger.info(f"  Device: {accelerator.device}")
        if accelerator.distributed_type:
            logger.info(f"  Distributed type: {accelerator.distributed_type}")
    
    # Start training
    trainor.start()
    
    # Clean up
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
