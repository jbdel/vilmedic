import torch
import copy
import os
import operator
import re
import json
import inspect

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.models import *
from vilmedic.datasets import *

from torch.optim import *
from torch.optim.lr_scheduler import *
from vilmedic.blocks.schedulers import LinearWarmupCosineAnnealingLR
import sys


def vilmedic_state_dict_versioning(params, version):
    params = {k.replace('module.', ''): v for k, v in params.items()}

    if version is None or version < '1.3.2':
        params = {k.replace('enc.0.cnn.', 'enc.model.'): v for k, v in params.items()}
        params = {k.replace('enc.1.weight', 'enc.visual_projection.weight'): v for k, v in params.items()}
        params = {k.replace('enc.1.bias', 'enc.visual_projection.bias'): v for k, v in params.items()}

    return params


def get_eval_func(models):
    dummy = models[0]
    if isinstance(dummy, nn.DataParallel):
        dummy = dummy.module
    assert hasattr(dummy, "eval_func")
    return dummy.eval_func


def create_optimizer(config, logger, model_params, state_dict=None):
    assert 'lr' in config.optim_params
    config.optim_params.lr = float(config.optim_params.lr)

    if hasattr(torch.optim, config.optimizer):
        optim = getattr(torch.optim, config.optimizer)
    else:
        raise NotImplementedError(config.optimizer)

    print(config.optim_params)
    optimizer = optim(model_params, **config.optim_params)
    logger.settings('Optimizer {} created'.format(type(optimizer).__name__))

    if state_dict is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
        logger.info('Optimizer state loaded')
    else:
        logger.info(optimizer)
    return optimizer


def create_model(config, dl, logger, from_training=True, state_dict=None):
    # Create model, give him dataloader also
    config = copy.deepcopy(config.model)
    model = eval(config.pop('proto'))(**config, dl=dl, logger=logger, from_training=from_training)
    logger.settings('Model {} created'.format(type(model).__name__))

    if state_dict is not None:
        if "model" not in state_dict:
            logger.critical('This checkpoint is not valid. Key "model" is missing from dict.')
            sys.exit()
        params = vilmedic_state_dict_versioning(state_dict["model"], state_dict.get('__version__', None))
        model.load_state_dict(params, strict=True)
        logger.info('Model state loaded')
    else:
        logger.info(model)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    return model


def create_data_loader(config, split, logger, called_by_validator=False, called_by_ensemblor=False):
    dataset_config = copy.deepcopy(config.dataset)
    # Its important the dataset receive info if call from ensemblor (test time):
    # split can be train with validation transformation
    dataset = eval(dataset_config.proto)(split=split,
                                         ckpt_dir=config.ckpt_dir,
                                         called_by_ensemblor=called_by_ensemblor,
                                         **dataset_config)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    # RandomSampler for train split, during training only
    if split == 'train' and not called_by_validator:
        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=config.batch_size,
            drop_last=config.drop_last or False)
        logger.info('Using ' + type(sampler.sampler).__name__)

    else:
        sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=config.batch_size,
            drop_last=False)

    # Print dataset for training and ensemblor
    if not called_by_validator or called_by_ensemblor:
        logger.settings('DataLoader')
        logger.info(dataset)

    return DataLoader(dataset,
                      num_workers=dataset_config.num_workers or 4,
                      collate_fn=collate_fn,
                      batch_sampler=sampler,
                      pin_memory=True)


def create_scaler(config, logger, state_dict=None):
    scaler = torch.cuda.amp.GradScaler(enabled=(config.use_amp or False))
    logger.settings('Using scaler : {}'.format(scaler.is_enabled()))
    if state_dict is not None and "scaler" in state_dict:
        scaler.load_state_dict(state_dict["scaler"])
        logger.info('Scaler state loaded')
    return scaler


def create_training_scheduler(config, optimizer, logger, state_dict=None):
    config = copy.deepcopy(config)
    training_scheduler = TrainingScheduler(lr_decay_func=config.lr_decay,
                                           optimizer=optimizer,
                                           early_stop_metric=config.early_stop_metric,
                                           early_stop_limit=config.early_stop,
                                           lr_decay_params=config.lr_decay_params)
    logger.settings('Training scheduler created')
    if state_dict is not None and "training_scheduler" in state_dict:
        training_scheduler.load_state_dict(state_dict["training_scheduler"])
        logger.info('Training scheduler state loaded')
    else:
        logger.info(training_scheduler)
    return training_scheduler


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
                'Resuming checkpoint after epoch {} with tag {}.'.format(self.current_epoch + 1, self.current_tag))

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


class TrainingScheduler(object):
    iter_step_scheduler = {"CyclicLR", "OneCycleLR", "CosineAnnealingWarmRestarts"}
    epoch_step_scheduler = {"LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR", "ConstantLR", "LinearLR",
                            "ExponentialLR", "ChainedScheduler", "SequentialLR", "CosineAnnealingLR",
                            "LinearWarmupCosineAnnealingLR"}
    val_step_scheduler = {"ReduceLROnPlateau"}

    def __init__(self, lr_decay_func, optimizer, early_stop_metric, early_stop_limit, lr_decay_params):
        super().__init__()

        self.epoch = 0
        self.early_stop = 0
        self.early_stop_limit = early_stop_limit
        self.metric_comp_func = operator.gt
        self.mode = 'max'
        self.current_best_metric = -float('inf')
        self.lr_decay_params = lr_decay_params
        self.early_stop_metric = early_stop_metric

        # 4info: You can decay_on_training_loss and have a early_stop_metric different than training loss
        self.decay_on_training_loss = self.lr_decay_params.decay_on_training_loss or False

        if early_stop_metric in ['validation_loss', 'training_loss']:
            self.metric_comp_func = operator.lt
            self.mode = 'min'
            self.current_best_metric = float('inf')

        self.scheduler_name = lr_decay_func
        if self.scheduler_name == 'ReduceLROnPlateau':
            self.lr_decay_params["mode"] = self.mode

        def remove_unused_args(func, **kwargs):
            sig = [param.name for param in inspect.signature(func).parameters.values()]
            return {k: v for k, v in kwargs.items() if k in sig}

        self.lr_decay_params = remove_unused_args(eval(lr_decay_func), **self.lr_decay_params)
        self.scheduler = eval(lr_decay_func)(optimizer, **self.lr_decay_params)

    def iteration_step(self):
        if self.scheduler_name in TrainingScheduler.iter_step_scheduler:
            self.scheduler.step()

    def epoch_step(self):
        self.epoch = self.epoch + 1
        if self.scheduler_name in TrainingScheduler.epoch_step_scheduler:
            self.scheduler.step()

    def eval_step(self, decay_metric=None, early_stop_score=None):
        ret = {
            "done_training": False,
            "save_state": False,
        }

        # LR scheduler
        if decay_metric is not None:
            if self.scheduler_name in TrainingScheduler.val_step_scheduler:
                self.scheduler.step(decay_metric)

        # Early stop
        if early_stop_score is not None:
            if self.metric_comp_func(early_stop_score, self.current_best_metric):
                self.current_best_metric = early_stop_score
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
        s += '    {0}: {1}\n'.format("decay_on_training_loss", self.decay_on_training_loss)
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
