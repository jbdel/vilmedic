import os
import json
import torch
import logging
from .utils import create_data_loader, get_eval_func
from vilmedic.scorers.scores import compute_scores
from vilmedic.scorers.post_processing import post_processing


class InitValidator(object):
    def __init__(self, opts, models, seed):
        self.seed = seed
        self.opts = opts

        # Logger
        self.logger = logging.getLogger(str(seed))

        self.models = models
        self.metrics = opts.metrics
        self.post_processing = opts.post_processing
        self.epoch = 0


class Validator(InitValidator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        assert isinstance(self.models, list)
        self.scores = []
        self.models = [m.eval() for m in self.models]

        splits = [(split, create_data_loader(self.opts, split, self.opts.logger, called_by_validator=True))
                  for split in self.opts.splits]

        for split, dl in splits:
            self.logger.info('Running split: {} by ensembling {} models. '
                             'Using {}.'.format(split,
                                                len(self.models),
                                                type(dl.batch_sampler.sampler).__name__,
                                                ))

            eval_func = get_eval_func(self.models)
            with torch.no_grad():
                results = eval_func(self.models, self.opts, dl)

            # model must return at least loss or (refs and hyps)
            # TODO check refs and hyps together
            assert type(results) is dict and \
                   any(key in results for key in ['loss', 'refs', 'hyps']), \
                self.logger.error('Evaluation func does not return any evaluation keys')

            scores = dict()

            # Handle loss
            scores['loss'] = str(results.pop("loss", 0.0))

            # Handle metrics
            metrics = compute_scores(metrics=self.metrics,
                                     refs=results.pop('refs', None),
                                     hyps=results.pop('hyps', None),
                                     split=split,
                                     seed=self.seed,
                                     ckpt_dir=self.opts.ckpt_dir,
                                     epoch=self.epoch
                                     )
            scores.update(metrics)

            # Dumping things for potential post processing
            post_processing(post_processing=self.post_processing,
                            results=results,
                            split=split,
                            seed=self.seed,
                            ckpt_dir=self.opts.ckpt_dir,
                            epoch=self.epoch,
                            dl=dl
                            )

            # Logging scores
            self.logger.info(json.dumps(scores, indent=4, sort_keys=False))

            # Saving the metrics for current split
            self.scores.append(scores)
