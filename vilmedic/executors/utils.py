import torch
import copy
import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.networks import *
from vilmedic.datasets import *

from torch.optim import *


def create_optimizer(opts, logger, params):
    assert 'lr' in opts.optim_params
    optim = getattr(torch.optim, opts.optimizer)
    optimizer = optim(params, **opts.optim_params)

    logger.debug('Optimizer {} created'.format(type(optimizer).__name__))
    logger.info(optimizer)
    return optimizer


def create_model(opts, logger, state_dict=None):
    opts = copy.deepcopy(opts.model)
    model = eval(opts.proto)(**opts)

    # eval_func is the method called by the Validator to evaluate the model
    assert hasattr(model, "eval_func")

    logger.debug('Model {} created'.format(type(model).__name__))

    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict))
        logger.info('{} loaded.'.format(state_dict))
    else:
        logger.info(model)
    return model


def create_data_loader(opts, split, logger):
    if split == 'train':
        logger.debug('DataLoader')

    dataset_opts = copy.deepcopy(opts.dataset)
    dataset = eval(dataset_opts.proto)(split=split, ckpt_dir=opts.ckpt_dir, **dataset_opts)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    if split == 'train':
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
    def __init__(self, ckpt_dir, logger, seed):
        self.ckpt_dir = ckpt_dir
        self.seed = seed
        self.logger = logger
        self.current_tag = None
        self.current_step = None

    def save(self, model, tag, current_step):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.ckpt_dir, '{}_{}_{}.pth'.format(self.current_tag, self.current_step, self.seed))
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        path = os.path.join(self.ckpt_dir, '{}_{}_{}.pth'.format(tag, current_step, self.seed))
        torch.save(model, path)
        self.logger.info('{} saved.'.format(path))

        self.current_tag = str(tag)
        self.current_step = current_step
