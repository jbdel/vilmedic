import torch
import copy
import os
import operator

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.networks import *
from vilmedic.datasets import *

import torch_optimizer
from torch.optim import *
from torch_optimizer import *
import numpy as np
import torch.nn as nn
import re


def get_eval_func(models):
    dummy = models[0]
    if isinstance(dummy, nn.DataParallel):
        dummy = dummy.module
    return dummy.eval_func


def get_metric_comparison_func(early_stop_metric):
    func = operator.gt
    if early_stop_metric == 'loss':
        func = operator.lt
    return func


def create_optimizer(opts, logger, params):
    assert 'lr' in opts.optim_params
    if hasattr(torch.optim, opts.optimizer):
        optim = getattr(torch.optim, opts.optimizer)
    elif hasattr(torch_optimizer, opts.optimizer):
        optim = getattr(torch_optimizer, opts.optimizer)
    else:
        raise NotImplementedError(opts.optimizer)

    optimizer = optim(params, **opts.optim_params)

    logger.debug('Optimizer {} created'.format(type(optimizer).__name__))
    logger.info(optimizer)
    return optimizer


def create_model(opts, logger, state_dict=None):
    opts = copy.deepcopy(opts.model)
    model = eval(opts.pop('proto'))(**opts)

    # eval_func is the method called by the Validator to evaluate the model
    assert hasattr(model, "eval_func")

    logger.debug('Model {} created'.format(type(model).__name__))

    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict))
        logger.info('{} loaded.'.format(state_dict))
    else:
        logger.info(model)
    return model


def create_data_loader(opts, split, logger):
    dataset_opts = copy.deepcopy(opts.dataset)
    dataset = eval(dataset_opts.proto)(split=split, ckpt_dir=opts.ckpt_dir, **dataset_opts)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    if split == 'train':
        logger.debug('DataLoader')
        logger.info(dataset)

        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=opts.batch_size, drop_last=False)
        logger.info('Using' + type(sampler.sampler).__name__)

    else:  # eval or test
        sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=opts.batch_size, drop_last=False)

    return DataLoader(dataset,
                      num_workers=4,
                      collate_fn=collate_fn,
                      batch_sampler=sampler)


class CheckpointSaver(object):
    def __init__(self, ckpt_dir, logger, seed, ckpt=None):
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.logger = logger
        self.current_tag = None
        self.current_epoch = None

        if ckpt is not None:
            self.current_tag, self.current_epoch = self.extract_tag_and_step(ckpt)
            self.logger.debug(
                'Resuming checkpoint at epoch {} with tag {}.'.format(self.current_epoch, self.current_tag))

    def save(self, model, tag, current_epoch):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.ckpt_dir,
                                    '{}_{}_{}.pth'.format(self.current_tag, self.current_epoch, self.seed))
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        tag = np.round(tag, 6)
        path = os.path.join(self.ckpt_dir, '{}_{}_{}.pth'.format(tag, current_epoch, self.seed))
        torch.save(model, path)
        self.logger.info('{} saved.'.format(path))

        self.current_tag = tag
        self.current_epoch = current_epoch

    def extract_tag_and_step(self, ckpt):
        groups = re.match('.*/(.*?)_(.*?)_(.*?).pth', ckpt)
        return float(groups.group(1)), int(groups.group(2))
