import torch
import copy
import os
import operator
import re
import json
import inspect

import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.models import *
from vilmedic.datasets import *

from torch.optim import *
from torch.optim.lr_scheduler import *
from vilmedic.blocks.schedulers import LinearWarmupCosineAnnealingLR
import sys


def vilmedic_state_dict_versioning(params, version):
    params = {k.replace('module.', ''): v for k, v in params.items()}

    if version is None or version < '1.3.2':
        params = {k.replace('enc.0.cnn.', 'enc.model.'): v for k, v in params.items()}
        params = {k.replace('enc.1.weight', 'enc.visual_projection.weight'): v for k, v in params.items()}
        params = {k.replace('enc.1.bias', 'enc.visual_projection.bias'): v for k, v in params.items()}

    return params


def get_eval_func(models):
    dummy = models[0]
    if isinstance(dummy, nn.DataParallel):
        dummy = dummy.module
    assert hasattr(dummy, "eval_func")
    return dummy.eval_func


def create_optimizer(config, logger, model_params, state_dict=None):
    if not hasattr(config, 'optim_params') or not hasattr(config.optim_params, 'lr'):
        raise ValueError("config.optim_params.lr is required")
    
    # Convert to a proper dictionary
    # No need for type conversion since config parsing handles it now
    optim_dict = OmegaConf.to_container(config.optim_params, resolve=True)
    
    # Special handling for betas parameter (needs to be tuple)
    if 'betas' in optim_dict and isinstance(optim_dict['betas'], list):
        optim_dict['betas'] = tuple(optim_dict['betas'])

    if not hasattr(config, 'optimizer'):
        raise ValueError("config.optimizer is required")
    
    optimizer_name = config.optimizer
    if hasattr(torch.optim, optimizer_name):
        optim = getattr(torch.optim, optimizer_name)
    else:
        raise NotImplementedError(optimizer_name)

    optimizer = optim(model_params, **optim_dict)
    logger.settings('Optimizer {} created'.format(type(optimizer).__name__))

    if state_dict is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
        logger.info('Optimizer state loaded')
    else:
        logger.info(optimizer)
    return optimizer


def create_model(config, dl, logger, from_training=True, state_dict=None):
    # Create model, give him dataloader also
    config_copy = copy.deepcopy(config.model)
    
    # Get the prototype class name and remove it from the config
    if not hasattr(config_copy, 'proto'):
        raise ValueError("config.model.proto is required")
    
    proto = config_copy.get('proto')
    # Create a dictionary for the remaining config to pass as kwargs
    config_dict = {k: v for k, v in config_copy.items() if k != 'proto'}
    
    # Create the model instance
    model = eval(proto)(**config_dict, dl=dl, logger=logger, from_training=from_training)
    logger.settings('Model {} created'.format(type(model).__name__))

    if state_dict is not None:
        if "model" not in state_dict:
            logger.critical('This checkpoint is not valid. Key "model" is missing from dict.')
            sys.exit()
        params = vilmedic_state_dict_versioning(state_dict["model"], state_dict.get('__version__', None))
        model.load_state_dict(params, strict=True)
        logger.info('Model state loaded')
    else:
        logger.info(model)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        logger.warning("Using DataParallel - expect ~60-70% GPU utilization. "
                      "Consider using DistributedDataParallel (DDP) for better performance: "
                      "typically 85-95% GPU utilization and better scaling.")
    return model


def create_data_loader(config, split, logger, called_by_validator=False, called_by_ensemblor=False):
    dataset_config = copy.deepcopy(config.dataset)
    
    # Check if proto exists
    if not hasattr(dataset_config, 'proto'):
        raise ValueError("config.dataset.proto is required")
    
    # Get the proto class name
    proto = dataset_config.get('proto')
    
    # Its important the dataset receive info if call from ensemblor (test time):
    # split can be train with validation transformation
    dataset_kwargs = {k: v for k, v in dataset_config.items() if k != 'proto'}
    dataset = eval(proto)(split=split,
                         ckpt_dir=config.ckpt_dir,
                         called_by_ensemblor=called_by_ensemblor,
                         **dataset_kwargs)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    # Get batch size with fallback
    batch_size = config.get('batch_size', 1) if hasattr(config, 'batch_size') else 1
    
    # Get drop_last with fallback
    drop_last = config.get('drop_last', False) if hasattr(config, 'drop_last') else False
    
    # RandomSampler for train split, during training only
    if split == 'train' and not called_by_validator:
        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=batch_size,
            drop_last=drop_last)
        logger.info('Using ' + type(sampler.sampler).__name__)

    else:
        sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=False)

    # Print dataset for training and ensemblor
    if not called_by_validator or called_by_ensemblor:
        logger.settings('DataLoader')
        logger.info(dataset)

    # Get num_workers with fallback
    num_workers = dataset_config.get('num_workers', 4) if hasattr(dataset_config, 'num_workers') else 4
    
    # Optimize DataLoader for multi-GPU training
    persistent_workers = num_workers > 0  # Only if using workers
    prefetch_factor = 2 if num_workers > 0 else None  # Prefetch batches
    
    return DataLoader(dataset,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      batch_sampler=sampler,
                      pin_memory=True,
                      persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor)


def create_scaler(config, logger, state_dict=None):
    use_amp = config.get('use_amp', False) if hasattr(config, 'use_amp') else False
    logger.settings(f'Mixed precision (AMP): {"Enabled" if use_amp else "Disabled"}')

    # Use torch.amp.GradScaler for device-agnostic code (though currently CUDA-only)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    logger.settings('Using scaler : {}'.format(scaler.is_enabled()))
    if state_dict is not None and "scaler" in state_dict:
        scaler.load_state_dict(state_dict["scaler"])
        logger.info('Scaler state loaded')
    return scaler


def create_training_scheduler(config, optimizer, logger, state_dict=None):
    config_copy = copy.deepcopy(config)
    
    # Get required parameters with safety checks
    lr_decay = config_copy.get('lr_decay') if hasattr(config_copy, 'lr_decay') else None
    early_stop_metric = config_copy.get('early_stop_metric') if hasattr(config_copy, 'early_stop_metric') else None
    early_stop = config_copy.get('early_stop') if hasattr(config_copy, 'early_stop') else None
    lr_decay_params = config_copy.get('lr_decay_params') if hasattr(config_copy, 'lr_decay_params') else {}
    
    training_scheduler = TrainingScheduler(lr_decay_func=lr_decay,
                                          optimizer=optimizer,
                                          early_stop_metric=early_stop_metric,
                                          early_stop_limit=early_stop,
                                          lr_decay_params=lr_decay_params)
    logger.settings('Training scheduler created')
    if state_dict is not None and "training_scheduler" in state_dict:
        training_scheduler.load_state_dict(state_dict["training_scheduler"])
        logger.info('Training scheduler state loaded')
    else:
        logger.info(training_scheduler)
    return training_scheduler


class CheckpointSaver(object):
    def __init__(self, ckpt_dir, logger, seed, ckpt=None):
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.logger = logger
        self.current_tag = None
        self.current_epoch = None

        if ckpt is not None:
            self.current_tag, self.current_epoch = self.extract_tag_and_step(ckpt)
            self.logger.settings(
                'Resuming checkpoint after epoch {} with tag {}.'.format(self.current_epoch + 1, self.current_tag))

    def save(self, state_dict, tag, current_epoch):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.ckpt_dir,
                                    '{}_{}_{}.pth'.format(self.current_tag, self.current_epoch, self.seed))
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        tag = np.round(tag, 6)
        path = os.path.join(self.ckpt_dir, '{}_{}_{}.pth'.format(tag, current_epoch, self.seed))
        torch.save(state_dict, path)
        self.logger.info('{} saved.'.format(path))

        self.current_tag = tag
        self.current_epoch = current_epoch

    def extract_tag_and_step(self, ckpt):
        groups = re.match('.*/(.*?)_(.*?)_(.*?).pth', ckpt)
        return float(groups.group(1)), int(groups.group(2))


class LinearWarmupWrapper(object):
    """Wrapper for any scheduler to add linear warmup"""
    def __init__(self, scheduler, warmup_steps, base_lr):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
        self._warmup_complete = False
        
    def step(self, *args, **kwargs):
        """Step the scheduler, handling warmup if needed"""
        if self.current_step < self.warmup_steps:
            # During warmup, ignore the regular scheduler
            return
        elif not self._warmup_complete:
            # First step after warmup - ensure we're at base_lr
            self._warmup_complete = True
        
        # After warmup, delegate to wrapped scheduler
        return self.scheduler.step(*args, **kwargs)
    
    def get_lr(self):
        """Get current learning rates"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / float(self.warmup_steps)
            return [self.base_lr * warmup_factor for _ in self.scheduler.optimizer.param_groups]
        else:
            # Try to get lr from scheduler's get_lr method
            if hasattr(self.scheduler, 'get_lr'):
                try:
                    return self.scheduler.get_lr()
                except NotImplementedError:
                    # Some schedulers have get_lr but raise NotImplementedError
                    # Fall back to reading from optimizer param_groups
                    return [group['lr'] for group in self.scheduler.optimizer.param_groups]
            else:
                return [group['lr'] for group in self.scheduler.optimizer.param_groups]
    
    def state_dict(self):
        """Save state"""
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'warmup_complete': self._warmup_complete
        }
    
    def load_state_dict(self, state_dict):
        """Load state"""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_step = state_dict.get('current_step', 0)
        self._warmup_complete = state_dict.get('warmup_complete', False)


class TrainingScheduler(object):
    iter_step_scheduler = {"CyclicLR", "OneCycleLR", "CosineAnnealingWarmRestarts"}
    epoch_step_scheduler = {"LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR", "ConstantLR", "LinearLR",
                            "ExponentialLR", "ChainedScheduler", "SequentialLR", "CosineAnnealingLR",
                            "LinearWarmupCosineAnnealingLR"}
    val_step_scheduler = {"ReduceLROnPlateau"}

    def __init__(self, lr_decay_func, optimizer, early_stop_metric, early_stop_limit, lr_decay_params):
        super().__init__()

        # Initialize basic attributes
        self.epoch = 0
        self.early_stop = 0
        self.early_stop_limit = early_stop_limit
        self.early_stop_metric = early_stop_metric
        self.iteration_count = 0
        self.scheduler_name = lr_decay_func
        
        # Convert lr_decay_params to dictionary (values already converted to proper types in config parsing)
        if lr_decay_params is not None:
            if hasattr(lr_decay_params, 'items'):
                self.lr_decay_params = dict(lr_decay_params.items())
            else:
                self.lr_decay_params = lr_decay_params
        else:
            self.lr_decay_params = {}
        
        # Setup early stopping mode and comparison function
        if early_stop_metric in ['validation_loss', 'training_loss']:
            self.metric_comp_func = operator.lt
            self.mode = 'min'
            self.current_best_metric = float('inf')
        else:
            self.metric_comp_func = operator.gt
            self.mode = 'max'
            self.current_best_metric = -float('inf')
        
        # Handle decay_on_training_loss flag
        self.decay_on_training_loss = self.lr_decay_params.pop('decay_on_training_loss', False)
        
        # Set mode for ReduceLROnPlateau if not specified
        if self.scheduler_name == 'ReduceLROnPlateau' and 'mode' not in self.lr_decay_params:
            self.lr_decay_params['mode'] = self.mode
        
        # Extract warmup parameters
        self.warmup_steps = self.lr_decay_params.pop('warmup_steps', 0)
        self.warmup_ratio = self.lr_decay_params.pop('warmup_ratio', None)
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Create the base scheduler
        if lr_decay_func is not None:
            # Filter parameters to only those accepted by the scheduler
            self.lr_decay_params = self._filter_scheduler_params(lr_decay_func, self.lr_decay_params)
            base_scheduler = eval(lr_decay_func)(optimizer, **self.lr_decay_params)
        else:
            # Default scheduler that does nothing
            base_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        
        # Wrap with warmup if needed
        if self.warmup_steps > 0 or self.warmup_ratio is not None:
            self.scheduler = LinearWarmupWrapper(base_scheduler, self.warmup_steps, self.base_lr)
            self.use_warmup = True
        else:
            self.scheduler = base_scheduler
            self.use_warmup = False
    
    def _filter_scheduler_params(self, scheduler_name, params):
        """Filter parameters to only include those accepted by the scheduler."""
        scheduler_class = eval(scheduler_name)
        if scheduler_class is None:
            return {}
        sig = inspect.signature(scheduler_class).parameters
        return {k: v for k, v in params.items() if k in sig}

    def iteration_step(self, epoch_value=None):
        """
        Step the scheduler for per-iteration schedulers (e.g., CosineAnnealingWarmRestarts).
        Pass a fractional epoch value if provided."""
        self.iteration_count += 1
        
        # Handle warmup if enabled
        if self.use_warmup and hasattr(self.scheduler, 'current_step'):
            self.scheduler.current_step = self.iteration_count
            
            # Update learning rate during warmup
            if self.iteration_count <= self.scheduler.warmup_steps:
                lrs = self.scheduler.get_lr()
                for param_group, lr in zip(self.scheduler.scheduler.optimizer.param_groups, lrs):
                    param_group['lr'] = lr
        
        # Step iteration-based schedulers after warmup
        if self.scheduler_name in TrainingScheduler.iter_step_scheduler:
            if not self.use_warmup or self.iteration_count > self.scheduler.warmup_steps:
                if epoch_value is not None:
                    self.scheduler.step(epoch_value)
                else:
                    self.scheduler.step()

    def epoch_step(self):
        self.epoch = self.epoch + 1
        if self.scheduler_name in TrainingScheduler.epoch_step_scheduler:
            self.scheduler.step()

    def eval_step(self, decay_metric=None, early_stop_score=None):
        ret = {
            "done_training": False,
            "save_state": False,
        }

        # LR scheduler (only step after warmup is complete)
        if decay_metric is not None:
            if self.scheduler_name in TrainingScheduler.val_step_scheduler:
                if not self.use_warmup or self.iteration_count > self.scheduler.warmup_steps:
                    if self.use_warmup:
                        # Step the wrapped scheduler
                        self.scheduler.scheduler.step(decay_metric)
                    else:
                        self.scheduler.step(decay_metric)

        # Early stop
        if early_stop_score is not None:
            if self.metric_comp_func(early_stop_score, self.current_best_metric):
                self.current_best_metric = early_stop_score
                self.early_stop = 0
                ret["save_state"] = True
            else:
                self.early_stop += 1
                if self.early_stop == self.early_stop_limit:
                    ret["done_training"] = True
        return ret

    def __repr__(self):
        s = "TrainingScheduler (\n"
        s += self.scheduler_name + "\n"
        s += str(json.dumps(dict(self.lr_decay_params), indent=4, sort_keys=True)) + '\n'
        if self.use_warmup:
            s += 'Warmup Settings' + "\n"
            s += '    {0}: {1}\n'.format("warmup_steps", self.warmup_steps)
            s += '    {0}: {1}\n'.format("base_lr", self.base_lr)
        s += 'Early stopping' + "\n"
        s += '    {0}: {1}\n'.format("early_stop_limit", self.early_stop_limit)
        s += '    {0}: {1}\n'.format("metric_comp_func", self.metric_comp_func)
        s += '    {0}: {1}\n'.format("mode", self.mode)
        s += '    {0}: {1}\n'.format("current_best_metric", self.current_best_metric)
        s += '    {0}: {1}\n'.format("decay_on_training_loss", self.decay_on_training_loss)
        s += ')'
        return s

    def state_dict(self):
        training_sched = {key: value for key, value in self.__dict__.items() if key != 'scheduler'}
        training_sched["scheduler"] = self.scheduler.state_dict()
        return training_sched

    def load_state_dict(self, state_dict):
        if "scheduler" in state_dict:  # Retro compatible with older checkpoint version
            scheduler = state_dict.pop("scheduler")
            self.__dict__.update(state_dict)
            # Handle warmup wrapper if present
            if self.use_warmup and hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(scheduler)
            elif hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(scheduler)
