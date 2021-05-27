import numpy as np
import random
import torch
import tqdm
import sys

from .base import Base
from .validator import Validator
from .utils import CheckpointSaver, create_model, create_data_loader
import time

class InitTrainor(Base):
    def __init__(self, opts):
        super().__init__(opts)
        self.seed = self.set_seed()

        # Checkpoints
        self.saver = CheckpointSaver(root=self.ckpt_dir, seed=self.seed)

        # Dataloader
        self.dl = create_data_loader(self.opts, 'train', self.ckpt_dir)

        # Model
        self.model = create_model(self.opts)
        # self.model = torch.nn.DataParallel(self.model)
        assert hasattr(self.model, "eval_func")
        self.model.cuda()
        print('\033[1m\033[91mModel {} created \033[0m'.format(type(self.model).__name__))
        print(self.model)

        # Optimizer
        self.optimizer = self.create_optimizer()

        # Hyper
        self.lr = self.opts.lr
        self.early_stop_metric = self.opts.early_stop_metric
        self.eval_start = self.opts.eval_start
        self.grad_accu = self.opts.grad_accu
        if self.grad_accu is None:
            self.grad_accu = 1

        # Validator is None at init
        self.evaluator: Validator = None

    @staticmethod
    def generate_seed():
        return int(repr(round(time.time() * 1000))[-7:])

    def set_seed(self, seed=None):
        if seed is None:
            seed = InitTrainor.generate_seed()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return seed

    def create_optimizer(self):
        if self.opts.optimizer is not None:
            optimizer = self.opts.optimizer
        else:
            optimizer = 'adam'

        params = self.model.parameters()

        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.opts.lr, weight_decay=self.opts.weight_decay)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.lr, weight_decay=self.opts.weight_decay)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))

        return optimizer


class Trainor(InitTrainor):
    def __init__(self, opts):
        super().__init__(opts=opts)

    def start(self):
        lr_patience = 0
        early_stop = 0

        for epoch in range(0, self.opts.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            iteration = 0
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            for batch in pbar:
                out = self.model(**batch)
                loss = out['loss']
                # logits = out['logits']
                # #
                # ma = torch.argmax(logits, dim=-1)
                # print(ma)
                # print(ma)
                # print(self.dl.dataset.tgt_tokenizer.decode(ma, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                # # ma = torch.argmax(logits, dim=-1)[1]
                # # print(self.dl.dataset.tgt_tokenizer.decode(ma, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                # print("#####")

                loss.backward()
                iteration += 1
                losses.append(loss.item())

                if iteration % self.grad_accu == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    pbar.set_description(
                        'Epoch {}, Lr {}, Loss {:.2f}, {} {:.2f}, ES {}'.format(
                            epoch + 1,
                            self.lr,
                            sum(losses) / iteration,
                            self.early_stop_metric,
                            self.evaluator.mean_eval_metric,
                            early_stop,
                        ))
                # break
                # if iteration >= 1:
                #     break
                # self.model.eval()
                # print(self.model.enc_dec.generate(batch['input_ids'].cuda()))
                # self.model.train()

            # Perform last update if needed
            if iteration % self.grad_accu != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

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
            print("Early stopped reached")
            sys.exit()
