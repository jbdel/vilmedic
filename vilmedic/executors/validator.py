import os
import torch
from .base import Base
from .utils import create_data_loader
from vilmedic.scorers.scores import compute_scores


class InitValidator(Base):
    def __init__(self, opts, models, seed):
        super().__init__(opts)
        self.models = models
        self.metrics = opts.metrics
        self.seed = seed

        self.mean_eval_metric = 0.0
        self.epoch = 0
        os.makedirs(self.ckpt_dir, exist_ok=True)


class Validator(InitValidator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        assert isinstance(self.models, list)
        self.models = [m.eval() for m in self.models]
        self.scores = []

        splits = [(split, create_data_loader(self.opts, split, self.ckpt_dir))
                  for split in self.opts.splits]

        for split, dl in splits:
            print('Running split: {} by ensembling {} models. '
                  'Using {}.'.format(split,
                                     len(self.models),
                                     type(dl.batch_sampler.sampler).__name__,
                                     ))
            self.split = split
            self.dl = dl
            eval_func = self.models[0].eval_func
            with torch.no_grad():
                self.losses, self.refs, self.hyps = eval_func(self.models, self.opts, self.dl)

            # Handle scores
            scores = compute_scores(metrics=self.metrics,
                                    refs=self.refs,
                                    hyps=self.hyps,
                                    split=self.split,
                                    seed=self.seed,
                                    ckpt_dir=self.ckpt_dir,
                                    epoch=self.epoch
                                    )

            print(scores)
            self.scores.append(scores)
