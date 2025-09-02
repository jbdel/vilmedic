"""
Accelerate-enabled Trainor for distributed training.
This module provides a modified version of the Trainor that uses Hugging Face Accelerate
for efficient multi-GPU training.
"""

import numpy as np
import logging
import torch
import tqdm
import sys
import os
from vilmedic import __version__

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .validator import Validator
from .validator_accelerate import ValidatorAccelerate
from .utils import (CheckpointSaver, create_model, create_data_loader, 
                    create_optimizer, create_training_scheduler, get_safe_logger)


class ConfigTrainorAccelerate(object):
    def __init__(self, config, seed, accelerator: Accelerator):
        # Store accelerator
        self.accelerator = accelerator
        
        # Misc
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
        
        # Mixed precision is handled by accelerator
        self.use_amp = config.get('use_amp', False)
        
        # Do we resume training?
        if self.ckpt is not None:
            self.state = torch.load(self.ckpt, map_location='cpu')
        
        # Setup logger - main process gets real logger, others get a safe dummy
        if self.accelerator.is_main_process:
            self.logger = logging.getLogger(str(self.seed))
            component_logger = self.logger  # Main process uses real logger
        else:
            self.logger = None
            component_logger = get_safe_logger(None, False)  # Others get dummy logger
        
        # Checkpointer only on main process
        self.saver = CheckpointSaver(
            ckpt_dir=self.ckpt_dir,
            logger=self.logger,  # Real logger or None
            seed=self.seed,
            ckpt=self.ckpt
        ) if self.accelerator.is_main_process else None
        
        # Create components - all processes need these
        self.dl = create_data_loader(
            self.config,
            split='train',
            logger=component_logger,  # Real for main, dummy for others
            from_accelerate=True
        )
        
        self.model = create_model(
            self.config,
            dl=self.dl,
            logger=component_logger,  # Real for main, dummy for others
            from_training=True,
            state_dict=self.state,
            from_accelerate=True
        )
        
        self.optimizer = create_optimizer(
            config=self.config,
            logger=component_logger,  # Real for main, dummy for others
            model_params=self.model.parameters(),
            state_dict=self.state
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.dl = self.accelerator.prepare(
            self.model, self.optimizer, self.dl
        )
        
        # Create scheduler after prepare
        self.training_scheduler = create_training_scheduler(
            config=self.config,
            optimizer=self.optimizer,
            logger=component_logger,  # Real for main, dummy for others
            state_dict=self.state
        )
        
        # Validator is None at init (will be set from outside on main process)
        self.evaluator: Validator = None


class TrainorAccelerate(ConfigTrainorAccelerate):
    def __init__(self, config, seed, accelerator: Accelerator):
        super().__init__(config=config, seed=seed, accelerator=accelerator)
    
    def _train_one_epoch(self, epoch):
        """Train for one epoch and return losses."""
        self.model.train()
        losses = []
        log = ""
        
        # Progress bar only on main process
        pbar = tqdm.tqdm(self.dl, total=len(self.dl)) if self.accelerator.is_main_process else self.dl
        
        for iteration, batch in enumerate(pbar, start=1):
            with self.accelerator.accumulate(self.model):
                out = self.model(**batch, epoch=epoch, iteration=iteration)        
                loss = out['loss']
                
                # Skip NaN/Inf losses
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.accelerator.is_main_process and self.logger:
                        self.logger.warning(f"NaN/Inf loss detected at epoch {epoch+1}, iteration {iteration}. Skipping...")
                    self.optimizer.zero_grad()
                    continue
                
                self.accelerator.backward(loss)
                
                # Gradient clipping when synced
                if self.accelerator.sync_gradients and self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Gather and log loss
                gathered_loss = self.accelerator.gather(loss.detach())
                if self.accelerator.is_main_process:
                    losses.append(gathered_loss.mean().item())
                
            # Update scheduler and log when gradients sync
            if self.accelerator.sync_gradients:
                frac_epoch = epoch + float(iteration) / len(self.dl)
                self.training_scheduler.iteration_step(frac_epoch)
                
                if self.accelerator.is_main_process and losses:
                    avg_loss = sum(losses) / len(losses)
                    log = self._format_log(epoch, avg_loss, out)
                    pbar.set_description(log)
            break
        return losses, log
    
    def _format_log(self, epoch, avg_loss, out):
        """Format training log string."""
        return 'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {} {}'.format(
            epoch + 1,
            [param_group['lr'] for param_group in self.optimizer.param_groups],
            avg_loss,
            self.training_scheduler.early_stop_metric,
            self.training_scheduler.current_best_metric,
            self.training_scheduler.early_stop,
            out.get("custom_print", "")
        )
    
    def _run_evaluation(self, epoch):
        """Run evaluation - distributed for multi-GPU, single for single-GPU."""
        if self.evaluator is None:
            return None
        
        # Check if we're using ValidatorAccelerate (multi-GPU)
        if isinstance(self.evaluator, ValidatorAccelerate):
            # Distributed evaluation - all processes participate
            self.evaluator.epoch = epoch
            self.evaluator.start()  # This handles model.eval() internally
            # Only main process has scores
            return self.evaluator.scores if self.accelerator.is_main_process else None
        else:
            # Single GPU evaluation (regular Validator)
            if self.accelerator.is_main_process:
                self.model.eval()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.evaluator.models = [unwrapped_model]
                self.evaluator.epoch = epoch
                self.evaluator.start()
                self.model.train()
                return self.evaluator.scores
            else:
                return None
    
    def _save_checkpoint(self, epoch, early_stop_score):
        """Save model checkpoint."""
        if not self.saver:
            return
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.saver.save(
            state_dict={
                "model": unwrapped_model.state_dict(),
                "training_scheduler": self.training_scheduler.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "__version__": __version__,
                "accelerate_state": {
                    "epoch": epoch + 1,
                    "num_processes": self.accelerator.num_processes,
                    "mixed_precision": str(self.accelerator.mixed_precision),
                }
            },
            tag=early_stop_score,
            current_epoch=epoch + 1,
        )
    
    def start(self):
        """Main training loop with Accelerate integration."""
        
        for epoch in range(int(self.training_scheduler.epoch), self.config.epochs + 1):
            # Train for one epoch
            losses, log = self._train_one_epoch(epoch)
            
            # Synchronize and log
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.logger and log:
                    self.logger.info(log)
                self.training_scheduler.epoch_step()
            
            # Check if we should evaluate/checkpoint
            do_eval = epoch + 1 >= self.eval_start
            do_earl_stop = epoch + 1 >= self.early_stop_start
            do_lr_decay = epoch + 1 >= self.decay_metric_start
            training_loss = sum(losses) / len(losses) if losses else float('inf')
            
            # Run evaluation (all processes participate if using ValidatorAccelerate)
            scores = None
            if do_eval and self.evaluator is not None:
                scores = self._run_evaluation(epoch)
            
            # Main process handles metrics and checkpointing
            if self.accelerator.is_main_process:
                early_stop_score = None
                decay_metric = None
                early_stop_metric = self.config.get('early_stop_metric') if hasattr(self.config, 'early_stop_metric') else None
                
                # Set early stop score from training loss if applicable
                if early_stop_metric == "training_loss" and do_earl_stop:
                    early_stop_score = training_loss
                
                # Use evaluation scores if available
                if scores and early_stop_metric != "training_loss" and do_earl_stop:
                    early_stop_score = np.mean([s[early_stop_metric] for s in scores])
                
                # Determine decay metric
                if do_lr_decay:
                    decay_metric = training_loss if self.training_scheduler.decay_on_training_loss else early_stop_score
                
                # Update scheduler and check for early stopping
                ret = self.training_scheduler.eval_step(decay_metric=decay_metric, early_stop_score=early_stop_score)
                
                if ret["done_training"]:
                    if self.logger:
                        self.logger.info("Early stopping reached")
                    self.accelerator.wait_for_everyone()
                    sys.exit()
                
                if ret["save_state"]:
                    self._save_checkpoint(epoch, early_stop_score)
            
            # Ensure all processes are synchronized before next epoch
            self.accelerator.wait_for_everyone()
        
        # Training completed
        if self.accelerator.is_main_process and self.logger:
            self.logger.info("Training completed successfully!")
