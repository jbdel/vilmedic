"""
Accelerate-enabled Validator for distributed evaluation.
Handles multi-GPU validation with proper order preservation and result gathering.
"""

import torch
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from accelerate import Accelerator
from .utils import create_data_loader, get_eval_func, get_safe_logger
from vilmedic.blocks.scorers.scores import compute_scores
from vilmedic.blocks.scorers.post_processing import post_processing
from omegaconf import ListConfig


class ValidatorAccelerate:
    """Validator that handles distributed evaluation using Accelerate."""
    
    def __init__(self, config, model, train_dl, seed, from_training, accelerator: Accelerator):
        """
        Initialize ValidatorAccelerate.
        
        Args:
            config: Validation configuration
            model: Model to evaluate (already wrapped by accelerator)
            train_dl: Training dataloader (for reference)
            seed: Random seed
            from_training: Whether called from training
            accelerator: Accelerate instance for distributed processing
        """
        self.accelerator = accelerator
        self.seed = seed
        self.config = config
        self.from_training = from_training
        self.train_dl = train_dl
        self.epoch = 0
        
        # Keep the wrapped model
        self.model = model
        
        # Setup logger
        self.logger = logging.getLogger(str(seed)) if self.accelerator.is_main_process else get_safe_logger(None, False)
        
        # Metrics configuration
        self.metrics = config.get('metrics', []) if hasattr(config, 'metrics') else []
        if not isinstance(self.metrics, (list, ListConfig)):
            self.metrics = [self.metrics]
        
        # post_processing
        self.post_processing = config.get('post_processing', None) if hasattr(config, 'post_processing') else None
        
        # Get splits
        if not hasattr(config, 'splits'):
            if self.accelerator.is_main_process:
                self.logger.warning("No splits defined in config, using ['val'] as default")
            splits = ['val']
        else:
            splits = config.splits
        
        # Create and prepare validation dataloaders
        self.splits = []
        for split in splits:
            dl = create_data_loader(
                self.config,
                split,
                self.logger,
                called_by_validator=True,
                called_by_ensemblor=not from_training,
                from_accelerate=True
            )
            # Prepare dataloader for distributed processing
            dl_prepared = self.accelerator.prepare(dl)
            self.splits.append((split, dl_prepared))
    
    def start(self):
        """Run distributed validation."""
        self.scores = []
        
        # Set model to eval mode on all processes
        self.model.eval()
        
        for split, dl in self.splits:
            if self.accelerator.is_main_process:
                self.logger.info(f'Running distributed validation on split: {split} '
                               f'with {self.accelerator.num_processes} processes')
            
            # Run distributed evaluation
            results = self._distributed_eval(split, dl)
            
            # Only main process computes final metrics
            if self.accelerator.is_main_process:
                scores = self._compute_metrics(results, split)
                self.scores.append(scores)
            
            # Synchronize all processes
            self.accelerator.wait_for_everyone()
        
        # Set model back to train mode
        self.model.train()
    
    def _distributed_eval(self, split: str, dl) -> Dict[str, Any]:
        """
        Run evaluation across all GPUs and gather results.
        Each process evaluates its portion of data.
        """
        # Get the unwrapped model for evaluation
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        eval_func = get_eval_func([unwrapped_model])
        
        with torch.no_grad():
            # Each process evaluates its portion of the dataloader
            local_results = eval_func(
                models=[unwrapped_model],
                config=self.config,
                dl=dl,
                from_training=self.from_training
            )
            
            # Extract results
            refs = local_results.get('refs', [])
            hyps = local_results.get('hyps', [])
            loss = local_results.get('loss', 0.0)
            
            # WORKAROUND: Ensure refs and hyps have the same length
            # Sometimes the model may fail to generate hypotheses for certain inputs
            # (e.g., due to beam search failures, max length constraints, or other issues)
            # This causes a mismatch between the number of references and hypotheses
            # TODO: Investigate why the model sometimes fails to generate outputs
            if refs and hyps and len(refs) != len(hyps):
                print(f"WARNING: Process {self.accelerator.process_index} has mismatched refs ({len(refs)}) and hyps ({len(hyps)})")
                # Pad the shorter list with empty strings to maintain alignment
                if len(hyps) < len(refs):
                    hyps = hyps + [''] * (len(refs) - len(hyps))
                elif len(refs) < len(hyps):
                    refs = refs + [''] * (len(hyps) - len(refs))
            
            # Calculate number of samples
            num_samples = len(refs) if refs else len(hyps) if hyps else 0
            total_loss = loss * num_samples if num_samples > 0 else 0.0
        
        # Gather results from all processes
        if self.accelerator.num_processes > 1:
            # Log local samples from ALL processes for debugging
            print(f"Process {self.accelerator.process_index}: refs={len(refs)}, hyps={len(hyps)}")
            
            # Ensure refs and hyps have the same length on each process
            if len(refs) != len(hyps):
                if self.accelerator.is_main_process:
                    self.logger.warning(f"Process {self.accelerator.process_index}: Mismatch between refs ({len(refs)}) and hyps ({len(hyps)})")
            
            # Use gather_for_metrics which properly handles uneven distributions
            # It automatically pads and then removes padding
            gathered_refs = self.accelerator.gather_for_metrics(refs)
            gathered_hyps = self.accelerator.gather_for_metrics(hyps)
            
            # Gather loss values
            loss_tensor = torch.tensor([total_loss], device=self.accelerator.device)
            samples_tensor = torch.tensor([num_samples], device=self.accelerator.device)
            gathered_loss = self.accelerator.gather(loss_tensor)
            gathered_samples = self.accelerator.gather(samples_tensor)
            
            # Only main process computes final metrics
            if self.accelerator.is_main_process:
                # Debug: Check gathered sizes
                self.logger.info(f"Gathered refs: {len(gathered_refs)}, Gathered hyps: {len(gathered_hyps)}")
                
                # Final safety check - ensure equal lengths
                if len(gathered_refs) != len(gathered_hyps):
                    self.logger.warning(f"Final mismatch after gathering: refs={len(gathered_refs)}, hyps={len(gathered_hyps)}")
                    # Trim to the shorter length to avoid assertion error
                    min_len = min(len(gathered_refs), len(gathered_hyps))
                    gathered_refs = gathered_refs[:min_len]
                    gathered_hyps = gathered_hyps[:min_len]
                    self.logger.warning(f"Trimmed to {min_len} samples")
                
                # Compute average loss
                total_loss_all = gathered_loss.sum().item()
                total_samples_all = gathered_samples.sum().item()
                avg_loss = total_loss_all / max(total_samples_all, 1)
                
                return {
                    'refs': gathered_refs,
                    'hyps': gathered_hyps,
                    'loss': avg_loss
                }
            else:
                return {}
        else:
            # Single process - return as is
            return {
                'refs': refs,
                'hyps': hyps,
                'loss': total_loss / max(num_samples, 1)
            }
    
    def _compute_metrics(self, results: Dict[str, Any], split: str) -> Dict[str, float]:
        """Compute metrics on the main process."""
        if not results:
            return {}
        
        scores = dict()
        
        # Handle loss
        scores['validation_loss'] = float(results.get("loss", 0.0))
        
        # Compute metrics
        refs = results.get('refs', None)
        hyps = results.get('hyps', None)
        
        if self.metrics and (refs is not None or hyps is not None):
            metrics = compute_scores(
                metrics=self.metrics,
                refs=refs,
                hyps=hyps,
                split=split,
                seed=self.seed,
                config=self.config,
                epoch=self.epoch,
                logger=self.logger
            )
            scores.update(metrics)
        
        # Post-processing if needed
        if self.post_processing is not None:
            # Create a results dict without refs/hyps for post-processing
            post_results = {k: v for k, v in results.items() if k not in ['refs', 'hyps', 'loss']}
            if post_results:
                post_processing(
                    post_processing=self.post_processing,
                    results=post_results,
                    split=split,
                    seed=self.seed,
                    ckpt_dir=self.config.get('ckpt_dir') if hasattr(self.config, 'ckpt_dir') else None,
                    epoch=self.epoch,
                    dl=None
                )
        
        # Log scores
        self.logger.info(json.dumps(scores, indent=4, sort_keys=False))
        
        return scores