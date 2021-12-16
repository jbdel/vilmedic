import collections
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
import re


def extract_seed_from_ckpt(ckpt):
    # 1.68_10_560435.pth to 560435
    assert os.path.exists(ckpt), '{} does not exist'.format(ckpt)
    return re.match(".*_(.*?).pth", ckpt).group(1)


def print_args(opts, splits, seed, override):
    logger = logging.getLogger(str(seed))
    logger.settings("Override dict")
    logger.info(json.dumps(OmegaConf.to_container(override), indent=4, sort_keys=True))

    for split in splits:
        d = OmegaConf.to_container(getattr(opts, split))
        logger.settings(split)
        logger.info(json.dumps(d, indent=4, sort_keys=True))


def get_args():
    """Return Omegaconf dict from command line
    Command line consists of one config and ovveriding args (others)"""

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args, others = parser.parse_known_args()

    # Get configs
    config = OmegaConf.load(args.config)
    includes = config.get("includes", [])

    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    # Loop over includes
    include_mapping = OmegaConf.create()
    for include in includes:
        if not os.path.exists(include):
            include = os.path.join(os.path.dirname(args.config), include)

        current_include_mapping = OmegaConf.load(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    # Override includes with current config
    config = OmegaConf.merge(include_mapping, config)

    # Override current config with additional args
    override = OmegaConf.from_dotlist(others)
    opts = OmegaConf.merge(config, override)

    return opts, override


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
