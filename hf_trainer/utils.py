import sys
import os
import logging
import math
import socket
import random
from typing import Optional, Tuple
from omegaconf import OmegaConf

# Add parent directory to path to import existing utilities
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import existing utilities from bin/utils.py
from bin.utils import (
    get_args,
    get,
    print_args,
    get_seed,
    extract_seed_from_ckpt,
    merge_with_dotlist,
    convert_numeric_strings
)
from bin.logger import set_logger


def setup_logger(ckpt_dir: str, seed: int, is_main_process: bool) -> logging.Logger:
    """Setup logger for training, handling both main and worker processes.
    
    Args:
        ckpt_dir: Directory for checkpoint/logs
        seed: Random seed (used as logger name)
        is_main_process: Whether this is the main process
    
    Returns:
        Configured logger instance
    """
    if is_main_process:
        set_logger(ckpt_dir, seed)
        logger = logging.getLogger(str(seed))
    else:
        logger = logging.getLogger(str(seed))
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.disabled = True
    return logger


def calculate_warmup_params(
    train_config,
    dataset_len: int,
    world_size: int = 1
) -> Tuple[int, int, int, int, int]:
    """Calculate training steps and warmup configuration.
    
    Args:
        train_config: Training configuration object
        dataset_len: Length of training dataset
        world_size: Number of processes for distributed training
    
    Returns:
        Tuple of (effective_batch_per_step, steps_per_epoch, total_training_steps, 
                  warmup_steps, warmup_ratio*100)
    """
    per_device_bs = int(train_config.get('batch_size', 8))
    grad_accu = int(train_config.get('grad_accu', 1))
    global_micro_batch = max(1, per_device_bs * max(1, world_size))
    effective_batch_per_step = max(1, global_micro_batch * max(1, grad_accu))

    steps_per_epoch = max(1, math.ceil(dataset_len / effective_batch_per_step))
    total_training_steps = steps_per_epoch * int(train_config.epochs)

    # Warmup heuristic: 3-10% depending on training length, minimum 100 steps
    if total_training_steps <= 2000:
        warmup_ratio = 0.10
    elif total_training_steps <= 10000:
        warmup_ratio = 0.06
    else:
        warmup_ratio = 0.03
    
    warmup_steps = max(100, int(round(total_training_steps * warmup_ratio)))
    warmup_steps = min(10000, warmup_steps)
    
    # Update config with calculated values
    if not hasattr(train_config, 'lr_decay_params') or train_config.lr_decay_params is None:
        train_config.lr_decay_params = OmegaConf.create({})
    train_config.lr_decay_params.num_warmup_steps = warmup_steps
    train_config.lr_decay_params.num_training_steps = total_training_steps
    
    return effective_batch_per_step, steps_per_epoch, total_training_steps, warmup_steps, int(warmup_ratio * 100)


def find_free_port(start_port: int = 29500, max_attempts: int = 100) -> int:
    """Find a free port for distributed training to avoid conflicts.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
    
    Returns:
        Available port number
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    
    # If no port found in range, return a random high port
    return random.randint(30000, 40000)


def get_world_size() -> int:
    """Get world size for distributed training.
    
    Returns:
        Number of processes (world size)
    """
    import torch
    
    world_size = 1
    try:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1
    
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
    except Exception:
        pass
    
    return world_size
