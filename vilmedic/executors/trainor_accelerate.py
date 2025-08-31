"""
Optimized Trainor using HuggingFace Accelerate for better multi-GPU performance.
Maintains full compatibility with existing vilmedic library structure.
"""

import numpy as np
import logging
import torch
import tqdm
import sys
import os
from vilmedic import __version__
from vilmedic.models import *

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate import DistributedType

from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader, create_optimizer, create_training_scheduler


class AcceleratedConfigTrainor(object):
    """Configuration and setup for Accelerate-based training."""
    
    def __init__(self, config, seed):
        # Initialize Accelerator with optimizations
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,  # Set to True if model has unused parameters
            broadcast_buffers=False  # Saves memory
        )
        
        # Configure Accelerator based on config
        mixed_precision = None
        if config.get('use_amp', False):
            # Determine precision based on config or default to fp16
            precision = config.get('amp_precision', 'fp16')
            if precision in ['fp16', 'bf16', 'fp8']:
                mixed_precision = precision
            else:
                mixed_precision = 'fp16'
        
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=config.get('grad_accu', 1),
            cpu=not torch.cuda.is_available(),
            kwargs_handlers=[ddp_kwargs],
            step_scheduler_with_optimizer=False,  # We handle scheduler manually
            project_dir=config.ckpt_dir if hasattr(config, 'ckpt_dir') else None,
        )
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Store config and seed
        self.config = config
        self.seed = seed
        self.state = None
        self.ckpt_dir = config.ckpt_dir
        self.ckpt = config.get('ckpt') if hasattr(config, 'ckpt') else None
        
        # Training parameters
        self.eval_start = config.get('eval_start', 0)
        self.decay_metric_start = config.get('decay_metric_start', 0)
        self.early_stop_start = config.get('early_stop_start', 0)
        self.grad_accu = config.get('grad_accu', 1)
        self.clip_grad_norm = config.get('clip_grad_norm') if hasattr(config, 'clip_grad_norm') else None
        
        # Resume training if checkpoint provided
        if self.ckpt is not None:
            self.state = torch.load(self.ckpt, map_location='cpu')
        
        # Logger - only on main process
        self.logger = logging.getLogger(str(self.seed))
        
        # Log Accelerator configuration
        if self.accelerator.is_main_process:
            self.logger.info(f"Accelerator initialized:")
            self.logger.info(f"  Distributed type: {self.accelerator.distributed_type}")
            self.logger.info(f"  Mixed precision: {self.accelerator.mixed_precision}")
            self.logger.info(f"  Num processes: {self.accelerator.num_processes}")
            self.logger.info(f"  Gradient accumulation steps: {self.grad_accu}")
            if self.accelerator.distributed_type != DistributedType.NO:
                self.logger.info(f"  Process index: {self.accelerator.process_index}")
                self.logger.info(f"  Local process index: {self.accelerator.local_process_index}")
        
        # Checkpoints (only on main process)
        self.saver = None
        if self.accelerator.is_main_process:
            self.saver = CheckpointSaver(
                ckpt_dir=self.ckpt_dir,
                logger=self.logger,
                seed=self.seed,
                ckpt=self.ckpt
            )
        
        # Create DataLoader
        self.dl = create_data_loader(
            self.config,
            split='train',
            logger=self.logger if self.accelerator.is_main_process else None
        )
        
        # Create Model - don't wrap with DataParallel since Accelerate handles it
        self.model = self._create_model_for_accelerate()
        
        # Create Optimizer
        self.optimizer = create_optimizer(
            config=self.config,
            logger=self.logger if self.accelerator.is_main_process else None,
            model_params=self.model.parameters(),
            state_dict=self.state
        )
        
        # Create LR Scheduler
        self.training_scheduler = create_training_scheduler(
            config=self.config,
            optimizer=self.optimizer,
            logger=self.logger if self.accelerator.is_main_process else None,
            state_dict=self.state
        )
        
        # Prepare model, optimizer, dataloader with Accelerate
        self.model, self.optimizer, self.dl = self.accelerator.prepare(
            self.model, self.optimizer, self.dl
        )
        
        # Handle scheduler preparation (some schedulers need special handling)
        if hasattr(self.training_scheduler, 'scheduler'):
            # Custom wrapper, prepare the inner scheduler
            if not isinstance(self.training_scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.training_scheduler.scheduler = self.accelerator.prepare(self.training_scheduler.scheduler)
        
        # Validator is None at init
        self.evaluator: Validator = None
        
        # Enable gradient checkpointing if requested
        gradient_checkpointing = config.get('gradient_checkpointing', False)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _create_model_for_accelerate(self):
        """Create model without DataParallel wrapping since Accelerate handles distribution."""
        import copy
        
        config_copy = copy.deepcopy(self.config.model)
        
        if not hasattr(config_copy, 'proto'):
            raise ValueError("config.model.proto is required")
        
        proto = config_copy.get('proto')
        config_dict = {k: v for k, v in config_copy.items() if k != 'proto'}
        
        # Create the model instance
        model = eval(proto)(**config_dict, dl=self.dl, logger=self.logger, from_training=True)
        
        if self.accelerator.is_main_process:
            self.logger.settings('Model {} created'.format(type(model).__name__))
        
        # Load state dict if provided
        if self.state is not None:
            if "model" not in self.state:
                if self.accelerator.is_main_process:
                    self.logger.critical('This checkpoint is not valid. Key "model" is missing from dict.')
                sys.exit()
            
            from .utils import vilmedic_state_dict_versioning
            params = vilmedic_state_dict_versioning(self.state["model"], self.state.get('__version__', None))
            model.load_state_dict(params, strict=True)
            
            if self.accelerator.is_main_process:
                self.logger.info('Model state loaded')
        elif self.accelerator.is_main_process:
            self.logger.info(model)
        
        # Move to device (Accelerate will handle the actual device placement)
        return model
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # Try to enable on the wrapped model
        model_to_check = self.model
        
        # If model has .model attribute (vilmedic wrapper), check the inner model
        if hasattr(model_to_check, 'model'):
            hf_model = model_to_check.model
            
            # Handle HuggingFace models
            if hasattr(hf_model, 'gradient_checkpointing_enable'):
                hf_model.gradient_checkpointing_enable()
                if self.accelerator.is_main_process:
                    self.logger.info('Gradient checkpointing enabled on HuggingFace model')
            # Handle encoder-decoder models
            elif hasattr(hf_model, 'encoder') and hasattr(hf_model.encoder, 'gradient_checkpointing_enable'):
                hf_model.encoder.gradient_checkpointing_enable()
                if hasattr(hf_model, 'decoder') and hasattr(hf_model.decoder, 'gradient_checkpointing_enable'):
                    hf_model.decoder.gradient_checkpointing_enable()
                if self.accelerator.is_main_process:
                    self.logger.info('Gradient checkpointing enabled on encoder-decoder')
            elif self.accelerator.is_main_process:
                self.logger.warning('Gradient checkpointing requested but not supported by the model')
        else:
            # Direct model
            if hasattr(model_to_check, 'gradient_checkpointing_enable'):
                model_to_check.gradient_checkpointing_enable()
                if self.accelerator.is_main_process:
                    self.logger.info('Gradient checkpointing enabled')
            elif self.accelerator.is_main_process:
                self.logger.warning('Gradient checkpointing requested but not supported by the model')


class AcceleratedTrainor(AcceleratedConfigTrainor):
    """Accelerate-optimized training loop."""
    
    def __init__(self, config, seed):
        super().__init__(config=config, seed=seed)
    
    def start(self):
        """Main training loop with Accelerate optimizations."""
        
        for epoch in range(int(self.training_scheduler.epoch), self.config.epochs + 1):
            self.model.train()
            
            losses = []
            log = ""
            
            # Create progress bar only on main process
            if self.accelerator.is_main_process:
                pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            else:
                pbar = self.dl
            
            # Training loop
            for iteration, batch in enumerate(pbar, start=1):
                # Accelerate handles device placement automatically
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    out = self.model(**batch, epoch=epoch, iteration=iteration)
                    
                    # If the model is taking care of its own training
                    if 'loss' not in out:
                        if self.accelerator.is_main_process:
                            pbar.set_description('Epoch {}, {}'.format(epoch + 1, out))
                        continue
                    
                    loss = out['loss']
                    
                    # Accelerate handles loss scaling for mixed precision
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping if specified
                    if self.clip_grad_norm is not None:
                        # Accelerate's clip_grad_norm_ handles unscaling automatically
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.clip_grad_norm
                        )
                    
                    # Optimizer step (Accelerate handles gradient accumulation)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Learning rate scheduler step (iteration-based)
                    if self.accelerator.sync_gradients:
                        frac_epoch = epoch + float(iteration) / len(self.dl)
                        self.training_scheduler.iteration_step(frac_epoch)
                
                # Gather loss across all processes for logging
                gathered_loss = self.accelerator.gather(loss.detach())
                if self.accelerator.is_main_process:
                    losses.append(gathered_loss.mean().item())
                    
                    # Update progress bar
                    if iteration % self.grad_accu == 0 and isinstance(pbar, tqdm.tqdm):
                        log = 'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {} {}'.format(
                            epoch + 1,
                            [param_group['lr'] for param_group in self.optimizer.param_groups],
                            sum(losses) / len(losses),
                            self.training_scheduler.early_stop_metric,
                            self.training_scheduler.current_best_metric,
                            self.training_scheduler.early_stop,
                            out.get("custom_print", "")
                        )
                        pbar.set_description(log)
            
            # End of epoch - only on main process
            if self.accelerator.is_main_process:
                self.logger.info(log if log else f"Epoch {epoch + 1} completed")
                
            # Epoch-based scheduler step
            self.training_scheduler.epoch_step()
            
            # Evaluation - only on main process
            if self.accelerator.is_main_process:
                early_stop_score = None
                decay_metric = None
                do_earl_stop = epoch + 1 >= self.early_stop_start
                do_lr_decay = epoch + 1 >= self.decay_metric_start
                do_eval = epoch + 1 >= self.eval_start
                training_loss = sum(losses) / len(losses) if losses else 0
                
                # Compute early_stop_score according to early_stop_metric
                early_stop_metric = self.config.get('early_stop_metric') if hasattr(self.config, 'early_stop_metric') else None
                if early_stop_metric == "training_loss" and do_earl_stop:
                    early_stop_score = training_loss
                
                # Do eval?
                if do_eval and self.evaluator is not None:
                    # Unwrap model for evaluation (get the original model)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    # Temporarily replace the model in evaluator
                    original_models = self.evaluator.models
                    self.evaluator.models = [unwrapped_model]
                    
                    self.evaluator.epoch = epoch
                    self.evaluator.start()
                    
                    # Restore original models
                    self.evaluator.models = original_models
                    
                    # Compute early stop according to evaluation metric
                    if early_stop_metric != "training_loss" and do_earl_stop:
                        early_stop_score = np.mean([s[early_stop_metric] for s in self.evaluator.scores])
                
                # Record decay_metric
                if do_lr_decay:
                    if self.training_scheduler.decay_on_training_loss:
                        decay_metric = training_loss
                    else:
                        decay_metric = early_stop_score
                
                # Check early stopping and save checkpoint
                ret = self.training_scheduler.eval_step(decay_metric=decay_metric, early_stop_score=early_stop_score)
                
                if ret["done_training"]:
                    self.logger.info("Early stopping reached")
                    self.accelerator.end_training()
                    sys.exit()
                    
                if ret["save_state"] and self.saver is not None:
                    # Save using Accelerate's unwrap to get the original model
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    # Get state dict
                    state_dict = {
                        "model": unwrapped_model.state_dict(),
                        "training_scheduler": self.training_scheduler.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "config": self.config,
                        "__version__": __version__,
                        "accelerator_state": {
                            "mixed_precision": self.accelerator.mixed_precision,
                            "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                        }
                    }
                    
                    self.saver.save(
                        state_dict=state_dict,
                        tag=early_stop_score,
                        current_epoch=epoch + 1,
                    )
            
            # Wait for all processes to complete the epoch
            self.accelerator.wait_for_everyone()
        
        # Training completed
        if self.accelerator.is_main_process:
            self.logger.info("Training completed!")
        
        self.accelerator.end_training()


# Convenience function to create the right trainor based on config
def create_trainor(config, seed, use_accelerate=True):
    """
    Create a trainor instance.
    
    Args:
        config: Training configuration
        seed: Random seed
        use_accelerate: If True, use AcceleratedTrainor, else use regular Trainor
    """
    if use_accelerate and torch.cuda.device_count() > 0:
        try:
            return AcceleratedTrainor(config, seed)
        except ImportError:
            logging.warning("Accelerate not installed, falling back to regular Trainor")
            from .trainor import Trainor
            return Trainor(config, seed)
    else:
        from .trainor import Trainor
        return Trainor(config, seed)


# For backward compatibility
Trainor = AcceleratedTrainor
