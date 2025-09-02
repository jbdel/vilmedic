import numpy as np
import logging
import torch
import tqdm
import sys
import os
from vilmedic import __version__

from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader, create_optimizer, create_training_scheduler, \
    create_scaler


class ConfigTrainor(object):
    def __init__(self, config, seed):
        # Misc
        self.config = config
        self.seed = seed
        self.state = None
        self.ckpt_dir = config.ckpt_dir
        self.ckpt = config.get('ckpt') if hasattr(config, 'ckpt') else None

        # Training
        self.eval_start = config.get('eval_start', 0)
        self.decay_metric_start = config.get('decay_metric_start', 0)
        self.early_stop_start = config.get('early_stop_start', 0)
        self.grad_accu = config.get('grad_accu', 1)
        self.clip_grad_norm = config.get('clip_grad_norm') if hasattr(config, 'clip_grad_norm') else None
        
        # Optional optimizations
        self.use_amp = config.get('use_amp', False) if hasattr(config, 'use_amp') else False
        # Do we resume training?
        if self.ckpt is not None:
            self.state = torch.load(self.ckpt)

        # Logger
        self.logger = logging.getLogger(str(self.seed))

        # Checkpoints
        self.saver = CheckpointSaver(ckpt_dir=self.ckpt_dir,
                                     logger=self.logger,
                                     seed=self.seed,
                                     ckpt=self.ckpt)

        # Dataloader
        self.dl = create_data_loader(self.config,
                                     split='train',
                                     logger=self.logger
                                     )

        # Model
        self.model = create_model(self.config,
                                  dl=self.dl,
                                  logger=self.logger,
                                  from_training=True,
                                  state_dict=self.state)
        

        # Optimizer
        self.optimizer = create_optimizer(config=self.config,
                                          logger=self.logger,
                                          model_params=self.model.parameters(),
                                          state_dict=self.state)

        # Lr Scheduler and early stop
        self.training_scheduler = create_training_scheduler(config=self.config,
                                                            optimizer=self.optimizer,
                                                            logger=self.logger,
                                                            state_dict=self.state)

        # Scaler
        self.scaler = create_scaler(config=self.config,
                                    logger=self.logger,
                                    state_dict=self.state)

        # Validator is None at init
        self.evaluator: Validator = None


class Trainor(ConfigTrainor):
    def __init__(self, config, seed):
        super().__init__(config=config, seed=seed)

    def start(self):

        for epoch in range(int(self.training_scheduler.epoch), self.config.epochs + 1):
            self.model.train()

            losses = []
            log = ""
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))

            # Training
            for iteration, batch in enumerate(pbar, start=1):
                # Use autocast with float16 when AMP is enabled
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    out = self.model(**batch, epoch=epoch, iteration=iteration)

                # If the model is taking care of it own training
                if 'loss' not in out:
                    pbar.set_description('Epoch {}, {}'.format(epoch + 1, out))
                    continue

                loss = out['loss']
                if isinstance(self.model, torch.nn.DataParallel):
                    loss = loss.mean()
                
                # Check for NaN or Inf loss and skip if found
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at epoch {epoch+1}, iteration {iteration}. Skipping...")
                    self.optimizer.zero_grad()  # Clear any accumulated gradients
                    continue
                
                self.scaler.scale(loss).backward()
                losses.append(loss.item())

                if iteration % self.grad_accu == 0:
                    # Only unscale when we're about to step
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    frac_epoch = epoch + float(iteration) / len(self.dl)
                    self.training_scheduler.iteration_step(frac_epoch)

                    # Calculate average loss only from valid iterations
                    avg_loss = sum(losses) / len(losses) if losses else float('nan')
                    log = 'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {} {}'.format(
                        epoch + 1,
                        [param_group['lr'] for param_group in self.optimizer.param_groups],
                        avg_loss,
                        self.training_scheduler.early_stop_metric,
                        self.training_scheduler.current_best_metric,
                        self.training_scheduler.early_stop,
                        out.get("custom_print", "")
                    )
                    pbar.set_description(log)

                # break
            # Perform last update if needed
            if (iteration % self.grad_accu != 0) and ('loss' in out):
                # Only unscale when we're about to step
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # ensure final scheduler step
                frac_epoch = epoch + float(iteration) / len(self.dl)
                self.training_scheduler.iteration_step(frac_epoch)

            # End of epoch
            self.logger.info(log)
            self.training_scheduler.epoch_step()

            # Evaluation starts
            early_stop_score = None
            decay_metric = None
            do_earl_stop = epoch + 1 >= self.early_stop_start
            do_lr_decay = epoch + 1 >= self.decay_metric_start
            do_eval = epoch + 1 >= self.eval_start
            # Calculate training loss only from valid (non-NaN) iterations
            training_loss = sum(losses) / len(losses) if losses else float('inf')

            # Compute early_stop_score according to early_stop_metric if specified
            early_stop_metric = self.config.get('early_stop_metric') if hasattr(self.config, 'early_stop_metric') else None
            if early_stop_metric == "training_loss" and do_earl_stop:
                early_stop_score = training_loss

            # Do eval ?
            if do_eval:
                self.evaluator.epoch = epoch
                self.evaluator.start()
                # Compute early stop according to evaluation metric if specified
                if early_stop_metric != "training_loss" and do_earl_stop:
                    early_stop_score = np.mean([s[early_stop_metric] for s in self.evaluator.scores])

            # Record decay_metric (will not be used further if scheduler != ReduceLROnPlateau)
            if do_lr_decay:
                if self.training_scheduler.decay_on_training_loss:
                    decay_metric = training_loss
                else:
                    decay_metric = early_stop_score

            ret = self.training_scheduler.eval_step(decay_metric=decay_metric, early_stop_score=early_stop_score)

            if ret["done_training"]:
                self.logger.info("Early stopped reached")
                sys.exit()
            if ret["save_state"]:
                self.saver.save(state_dict={"model": self.model.state_dict(),
                                            "training_scheduler": self.training_scheduler.state_dict(),
                                            "optimizer": self.optimizer.state_dict(),
                                            "config": self.config,
                                            "scaler": self.scaler.state_dict(),
                                            "__version__": __version__
                                            },
                                tag=early_stop_score,
                                current_epoch=epoch + 1,
                                )
