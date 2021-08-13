import argparse
import logging
import copy
import json
import time
import random
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from random import randrange


def print_args(opts, splits, seed):
    for split in splits:
        d = OmegaConf.to_container(getattr(opts, split))
        logger = logging.getLogger(str(seed))
        logger.debug(split)
        logger.info(json.dumps(d, indent=4, sort_keys=True))


def get_args():
    """Return Omegaconf dict from command line
    Command line consists of one config and ovveriding args (others)"""

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args, others = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    override = OmegaConf.from_dotlist(others)

    if override:
        print('Overriding dict:', override)

    opts = OmegaConf.merge(config, override)
    opts.ckpt_dir = os.path.join(opts.ckpt_dir, opts.name)
    os.makedirs(opts.ckpt_dir, exist_ok=True)
    return opts


def get(opts, mode):
    """ Create a personal dict for each executor (trainor, validator or ensemblor)
    For an executor, we give every args but the other executors args
    """
    exec_opts = copy.deepcopy(getattr(opts, mode))
    for att in list(opts.keys()):
        if att not in ['trainor', 'validator', 'ensemblor']:
            exec_opts[att] = opts[att]
    return exec_opts


def get_seed(seed=None):
    if seed is None:
        seed = randrange(100000, 999999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed
