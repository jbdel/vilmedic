import numpy as np
import logging
import torch
import tqdm
import sys
import os

from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader, create_optimizer, get_metric_comparison_func


class InitTrainor(object):
    def __init__(self, opts, seed):
        self.seed = seed
        self.opts = opts

        # Logger
        self.logger = logging.getLogger(str(seed))

        # Checkpoints
        self.saver = CheckpointSaver(ckpt_dir=self.opts.ckpt_dir, logger=self.logger, seed=self.seed,
                                     ckpt=self.opts.ckpt)

        # Dataloader
        self.dl = create_data_loader(self.opts, split='train', logger=self.logger)

        # Model
        self.model = create_model(self.opts, logger=self.logger, state_dict=self.opts.ckpt)
        self.model.cuda()

        # Optimizer
        self.optimizer = create_optimizer(opts=self.opts, logger=self.logger, params=self.model.parameters())

        # Training
        self.lr = self.optimizer.defaults['lr']
        self.eval_start = self.opts.eval_start
        self.grad_accu = self.opts.grad_accu or 1

        # Model selection
        self.early_stop_metric = self.opts.early_stop_metric
        self.metric_comp_func = get_metric_comparison_func(self.early_stop_metric)

        # Validator is None at init
        self.evaluator: Validator = None


class Trainor(InitTrainor):
    def __init__(self, opts, seed):
        super().__init__(opts=opts, seed=seed)

    def start(self):
        # setting up training parameters
        lr_patience = 0
        early_stop = 0
        start_epoch = 0
        self.evaluator.current_best_metric = 0.0 if self.metric_comp_func.__name__ == 'gt' else float('inf')

        # If we loaded a checkpoint, adjust parameters
        self.evaluator.current_best_metric = self.saver.current_tag or self.evaluator.current_best_metric
        start_epoch = self.saver.current_epoch or start_epoch

        # Training
        for epoch in range(int(start_epoch), self.opts.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            iteration = 0
            log = ""
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            for batch in pbar:
                if type(batch) is dict:
                    out = self.model(**batch)
                else:
                    out = self.model(batch)

                loss = out['loss']
                if isinstance(self.model, torch.nn.DataParallel):
                    loss = loss.mean()
                loss.backward()
                iteration += 1
                losses.append(loss.item())

                if iteration % self.grad_accu == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    log = 'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {}'.format(
                        epoch + 1,
                        self.lr,
                        sum(losses) / iteration,
                        self.early_stop_metric,
                        self.evaluator.current_best_metric,
                        early_stop,
                    )
                    pbar.set_description(log)

            # self.model.eval()
            # out = self.model(**batch)
            # logits = torch.argmax(out["logits"], dim=-1)
            # for l in logits:
            #     self.logger.info(self.dl.dataset.tokenizer.decode(l, skip_special_tokens=True,
            #                                                       clean_up_tokenization_spaces=False))
            # self.model.train()

            # Perform last update if needed
            if iteration % self.grad_accu != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger.info(log)

            if epoch > self.eval_start - 1:
                self.evaluator.epoch = epoch
                self.evaluator.start()

                # Fetch eval score and compute early stop
                mean_eval_metric = np.mean([s[self.early_stop_metric] for s in self.evaluator.scores])
                if self.metric_comp_func(mean_eval_metric, self.evaluator.current_best_metric):
                    self.saver.save(model=self.model.state_dict(), tag=mean_eval_metric, current_epoch=epoch)
                    self.evaluator.current_best_metric = mean_eval_metric
                    early_stop = 0
                    lr_patience = 0
                else:
                    early_stop += 1
                    lr_patience += 1
                    self.update_lr_plateau(lr_patience)
                    self.check_early_stop(early_stop)

    def update_lr_plateau(self, lr_patience):
        if self.lr == self.opts.lr_min:
            return

        # Apply decay if applicable
        if lr_patience % self.opts.lr_decay_patience == 0:
            lr = self.lr * self.opts.lr_decay_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def check_early_stop(self, early_stop):
        if early_stop == self.opts.early_stop:
            self.logger.info("Early stopped reached")
            sys.exit()
