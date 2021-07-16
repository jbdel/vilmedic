import numpy as np
import logging
import tqdm
import sys

from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader, create_optimizer


class InitTrainor(object):
    def __init__(self, opts, seed):
        self.seed = seed
        self.opts = opts

        # Logger
        self.logger = logging.getLogger(str(seed))

        # Checkpoints
        self.saver = CheckpointSaver(ckpt_dir=self.opts.ckpt_dir, logger=self.logger, seed=self.seed)

        # Dataloader
        self.dl = create_data_loader(self.opts, split='train', logger=self.logger)

        # Model
        self.model = create_model(self.opts, logger=self.logger)
        # self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

        # Optimizer
        self.optimizer = create_optimizer(opts=self.opts, logger=self.logger, params=self.model.parameters())

        # Hyper
        self.lr = self.optimizer.defaults['lr']
        self.early_stop_metric = self.opts.early_stop_metric
        self.eval_start = self.opts.eval_start
        self.grad_accu = self.opts.grad_accu or 1

        # Validator is None at init
        self.evaluator: Validator = None


class Trainor(InitTrainor):
    def __init__(self, opts, seed):
        super().__init__(opts=opts, seed=seed)

    def start(self):
        lr_patience = 0
        early_stop = 0

        for epoch in range(0, self.opts.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            iteration = 0
            log = ""
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            for batch in pbar:
                out = self.model(**batch)
                loss = out['loss']
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
                        self.evaluator.mean_eval_metric,
                        early_stop,
                    )
                    pbar.set_description(log)
                # break

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
                if mean_eval_metric > self.evaluator.mean_eval_metric:
                    self.saver.save(model=self.model.state_dict(), tag=mean_eval_metric, current_step=epoch)
                    self.evaluator.mean_eval_metric = mean_eval_metric
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
