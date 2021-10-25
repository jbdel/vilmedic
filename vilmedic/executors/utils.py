import torch
import copy
import os
import operator
import re
import json

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.networks import *
from vilmedic.datasets import *

import torch_optimizer
from torch.optim import *
from torch_optimizer import *
from torch.optim.lr_scheduler import *


def get_eval_func(models):
    dummy = models[0]
    if isinstance(dummy, nn.DataParallel):
        dummy = dummy.module
    return dummy.eval_func


def create_optimizer(opts, logger, params, state_dict=None):
    assert 'lr' in opts.optim_params
    if hasattr(torch.optim, opts.optimizer):
        optim = getattr(torch.optim, opts.optimizer)
    elif hasattr(torch_optimizer, opts.optimizer):
        optim = getattr(torch_optimizer, opts.optimizer)
    else:
        raise NotImplementedError(opts.optimizer)

    optimizer = optim(params, **opts.optim_params)
    logger.settings('Optimizer {} created'.format(type(optimizer).__name__))

    if state_dict is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
        logger.info('Optimizer state loaded')
    else:
        logger.info(optimizer)
    return optimizer


def create_model(opts, logger, state_dict=None):
    opts = copy.deepcopy(opts.model)
    model = eval(opts.pop('proto'))(**opts)
    logger.settings('Model {} created'.format(type(model).__name__))
    # eval_func is the method called by the Validator to evaluate the model
    assert hasattr(model, "eval_func")

    if state_dict is not None and "model" in state_dict:
        params = {k.replace('module.', ''): v for k, v in state_dict["model"].items()}
        model.load_state_dict(params)
        logger.info('Model state loaded')
    else:
        logger.info(model)

    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    return model.cuda()


def create_data_loader(opts, split, logger, called_by_validator=False):
    dataset_opts = copy.deepcopy(opts.dataset)
    dataset = eval(dataset_opts.proto)(split=split, ckpt_dir=opts.ckpt_dir, **dataset_opts)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    if split == 'train' and not called_by_validator:
        logger.settings('DataLoader')
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
            self.logger.settings(
                'Resuming checkpoint at epoch {} with tag {}.'.format(self.current_epoch, self.current_tag))

    def save(self, state_dict, tag, current_epoch):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.ckpt_dir,
                                    '{}_{}_{}.pth'.format(self.current_tag, self.current_epoch, self.seed))
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        tag = np.round(tag, 6)
        path = os.path.join(self.ckpt_dir, '{}_{}_{}.pth'.format(tag, current_epoch, self.seed))
        torch.save(state_dict, path)
        self.logger.info('{} saved.'.format(path))

        self.current_tag = tag
        self.current_epoch = current_epoch

    def extract_tag_and_step(self, ckpt):
        groups = re.match('.*/(.*?)_(.*?)_(.*?).pth', ckpt)
        return float(groups.group(1)), int(groups.group(2))


def create_training_scheduler(opts, optimizer, logger, state_dict=None):
    opts = copy.deepcopy(opts)
    training_scheduler = TrainingScheduler(lr_decay_func=opts.lr_decay,
                                           optimizer=optimizer,
                                           early_stop_metric=opts.early_stop_metric,
                                           early_stop_limit=opts.early_stop,
                                           **opts.lr_decay_params)
    logger.settings('Training scheduler created')
    if state_dict is not None and "training_scheduler" in state_dict:
        training_scheduler.load_state_dict(state_dict["training_scheduler"])
        logger.info('Training scheduler state loaded')
    else:
        logger.info(training_scheduler)
    return training_scheduler


class TrainingScheduler(object):
    def __init__(self, lr_decay_func, optimizer, early_stop_metric, early_stop_limit, **lr_decay_params):
        super().__init__()

        self.epoch = 0
        self.early_stop = 0
        self.early_stop_limit = early_stop_limit
        self.metric_comp_func = operator.gt
        self.mode = 'max'
        self.current_best_metric = 0.0
        self.lr_decay_params = lr_decay_params

        if early_stop_metric == 'loss':
            self.metric_comp_func = operator.lt
            self.mode = 'min'
            self.current_best_metric = float('inf')

        self.scheduler_name = lr_decay_func
        if self.scheduler_name == 'ReduceLROnPlateau':
            lr_decay_params["mode"] = self.mode

        self.scheduler = eval(lr_decay_func)(optimizer, **lr_decay_params)

    def step(self, mean_eval_metric=None, training_loss=None):
        ret = {
            "done_training": False,
            "save_state": False,
        }
        self.epoch = self.epoch + 1

        # If eval has not started, dont compute early stop
        if mean_eval_metric is None:
            return ret

        # LR sched
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(mean_eval_metric)
        else:
            self.scheduler.step()

        # Early stop
        if self.metric_comp_func(mean_eval_metric, self.current_best_metric):
            self.current_best_metric = mean_eval_metric
            self.early_stop = 0
            ret["save_state"] = True
        else:
            self.early_stop += 1
            if self.early_stop == self.early_stop_limit:
                ret["done_training"] = True
        return ret

    def __repr__(self):
        s = "TrainingScheduler (\n"
        s += self.scheduler_name + "\n"
        s += str(json.dumps(dict(self.lr_decay_params), indent=4, sort_keys=True)) + '\n'
        s += 'Early stopping' + "\n"
        s += '    {0}: {1}\n'.format("early_stop_limit", self.early_stop_limit)
        s += '    {0}: {1}\n'.format("metric_comp_func", self.metric_comp_func)
        s += '    {0}: {1}\n'.format("mode", self.mode)
        s += '    {0}: {1}\n'.format("current_best_metric", self.current_best_metric)
        s += ')'
        return s

    def state_dict(self):
        training_sched = {key: value for key, value in self.__dict__.items() if key != 'scheduler'}
        training_sched["scheduler"] = self.scheduler.state_dict()
        return training_sched

    def load_state_dict(self, state_dict):
        if "scheduler" in state_dict:  # Retro compatible with older checkpoint version
            scheduler = state_dict.pop("scheduler")
            self.__dict__.update(state_dict)
            self.scheduler.load_state_dict(scheduler)
