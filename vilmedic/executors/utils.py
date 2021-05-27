import torch
import copy
import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from vilmedic.networks import *
from vilmedic.datasets import *

def create_model(opts, state_dict=None):
    opts = copy.deepcopy(opts.model)
    model = eval(opts.proto)(**opts)
    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict))
        print(state_dict, 'loaded.')
    return model


def create_data_loader(opts, split, ckpt_dir):
    if split == 'train':
        print('\033[1m\033[91mDataLoader \033[0m')

    dataset_opts = copy.deepcopy(opts.dataset)
    dataset = eval(dataset_opts.proto)(split=split, ckpt_dir=ckpt_dir, **dataset_opts)

    if hasattr(dataset, 'get_collate_fn'):
        collate_fn = dataset.get_collate_fn()
    else:
        collate_fn = default_collate

    if split == 'train':
        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=opts.batch_size, drop_last=False)
        print('Using \033[1m\033[94m' + type(sampler.sampler).__name__ + '\033[0m')

    else:  # eval or test
        sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=opts.batch_size, drop_last=False)

    return DataLoader(dataset,
                      num_workers=4,
                      collate_fn=collate_fn,
                      batch_sampler=sampler)


class CheckpointSaver(object):
    def __init__(self, root, seed):
        self.root = root
        self.seed = seed
        self.current_tag = None
        self.current_step = None
        os.makedirs(self.root, exist_ok=True)

    def save(self, model, tag, current_step):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.root, '{}_{}_{}.pth'.format(self.current_tag, self.current_step, self.seed))
            assert os.path.exists(old_ckpt), old_ckpt
            os.remove(old_ckpt)

        path = os.path.join(self.root, '{}_{}_{}.pth'.format(tag, current_step, self.seed))
        torch.save(model, path)
        print('{} saved.'.format(path))

        self.current_tag = tag
        self.current_step = current_step
