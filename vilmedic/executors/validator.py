import os
import torch
import logging
from .utils import create_data_loader
from vilmedic.scorers.scores import compute_scores


class InitValidator(object):
    def __init__(self, opts, models, seed):
        self.seed = seed
        self.opts = opts

        # Logger
        self.logger = logging.getLogger(str(seed))

        self.models = models
        self.metrics = opts.metrics
        self.mean_eval_metric = 0.0

        self.epoch = 0


class Validator(InitValidator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        assert isinstance(self.models, list)
        self.scores = []
        self.models = [m.eval() for m in self.models]

        splits = [(split, create_data_loader(self.opts, split, self.opts.ckpt_dir))
                  for split in self.opts.splits]

        for split, dl in splits:
            self.logger.info('Running split: {} by ensembling {} models. '
                             'Using {}.'.format(split,
                                                len(self.models),
                                                type(dl.batch_sampler.sampler).__name__,
                                                ))

            eval_func = self.models[0].eval_func
            with torch.no_grad():
                losses, refs, hyps = eval_func(self.models, self.opts, dl)

            # Handle scores
            scores = compute_scores(metrics=self.metrics,
                                    refs=refs,
                                    hyps=hyps,
                                    split=split,
                                    seed=self.seed,
                                    ckpt_dir=self.opts.ckpt_dir,
                                    epoch=self.epoch
                                    )

            self.logger.info(scores)
            self.scores.append(scores)
