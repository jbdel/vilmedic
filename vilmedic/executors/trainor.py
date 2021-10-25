import numpy as np
import logging
import torch
import tqdm
import sys
import os

from torch.nn.utils import clip_grad_norm_
from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader, create_optimizer, create_training_scheduler


class InitTrainor(object):
    def __init__(self, opts, seed):
        self.seed = seed
        self.opts = opts
        self.state = None

        # Do we resume training?
        if opts.ckpt is not None:
            self.state = torch.load(opts.ckpt)

        # Logger
        self.logger = logging.getLogger(str(seed))

        # Checkpoints
        self.saver = CheckpointSaver(ckpt_dir=self.opts.ckpt_dir, logger=self.logger, seed=self.seed,
                                     ckpt=self.opts.ckpt)

        # Dataloader
        self.dl = create_data_loader(self.opts, split='train', logger=self.logger)

        # Model
        self.model = create_model(self.opts, logger=self.logger, state_dict=self.state)

        # Optimizer
        self.optimizer = create_optimizer(opts=self.opts, logger=self.logger, params=self.model.parameters(),
                                          state_dict=self.state)

        # Lr Scheduler and early stop
        self.training_scheduler = create_training_scheduler(opts=self.opts, optimizer=self.optimizer,
                                                            logger=self.logger, state_dict=self.state)

        # Training
        self.eval_start = self.opts.eval_start
        self.grad_accu = self.opts.grad_accu or 1
        self.clip_grad_norm = lambda x: clip_grad_norm_(x, self.opts.clip_grad_norm) if (
                self.opts.clip_grad_norm is not None) else lambda x: None

        # Validator is None at init
        self.evaluator: Validator = None


class Trainor(InitTrainor):
    def __init__(self, opts, seed):
        super().__init__(opts=opts, seed=seed)

    def start(self):

        for epoch in range(int(self.training_scheduler.epoch), self.opts.epochs + 1):
            self.model.train()

            losses = []
            log = ""
            do_eval = epoch > self.eval_start - 1
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))

            # Training
            for iteration, batch in enumerate(pbar, start=1):
                if type(batch) is dict:
                    out = self.model(**batch, dl=self.dl)
                else:
                    out = self.model(batch, self.dl)

                # If the model is taking care of it own training
                if 'loss' not in out:
                    pbar.set_description('Epoch {}, {}'.format(epoch + 1, out))
                    continue

                loss = out['loss']
                if isinstance(self.model, torch.nn.DataParallel):
                    loss = loss.mean()
                loss.backward()
                losses.append(loss.item())

                if iteration % self.grad_accu == 0:
                    self.clip_grad_norm(self.model.parameters())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    log = 'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {}'.format(
                        epoch + 1,
                        [param_group['lr'] for param_group in self.optimizer.param_groups],
                        sum(losses) / iteration,
                        self.opts.early_stop_metric,
                        self.training_scheduler.current_best_metric,
                        self.training_scheduler.early_stop,
                    )
                    pbar.set_description(log)
                # break

            # Perform last update if needed
            if (iteration % self.grad_accu != 0) and ('loss' in out):
                self.clip_grad_norm(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger.info(log)

            # Evaluation
            mean_eval_metric = None
            if do_eval:
                self.evaluator.epoch = epoch
                self.evaluator.start()
                mean_eval_metric = np.mean([s[self.opts.early_stop_metric] for s in self.evaluator.scores])

            # End of epochs instructions
            ret = self.training_scheduler.step(mean_eval_metric, training_loss=sum(losses) / iteration)
            if ret["done_training"]:
                self.logger.info("Early stopped reached")
                sys.exit()
            if ret["save_state"]:
                self.saver.save(state_dict={"model": self.model.state_dict(),
                                            "training_scheduler": self.training_scheduler.state_dict(),
                                            "optimizer": self.optimizer.state_dict()},
                                tag=mean_eval_metric,
                                current_epoch=epoch)
