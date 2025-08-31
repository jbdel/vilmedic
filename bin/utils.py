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
import yaml


def extract_seed_from_ckpt(ckpt):
    # 1.68_10_560435.pth to 560435
    assert os.path.exists(ckpt), '{} does not exist'.format(ckpt)
    return re.match(".*_(.*?).pth", ckpt).group(1)


def print_args(config, splits, seed, override):
    logger = logging.getLogger(str(seed))
    logger.settings("Override dict")
    logger.info(json.dumps(OmegaConf.to_container(override), indent=4, sort_keys=True))

    for split in splits:
        d = OmegaConf.to_container(getattr(config, split))
        logger.settings(split)
        logger.info(json.dumps(d, indent=4, sort_keys=True))


def convert_numeric_strings(obj):
    """
    Recursively convert string representations of numbers to actual numbers.
    Handles scientific notation (e.g., "1e-8") and regular numbers.
    """
    if isinstance(obj, str):
        # Try to convert to number
        # Check if it looks like a number (including scientific notation)
        if re.match(r'^-?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?$', obj.strip()):
            try:
                # Try integer first
                if '.' not in obj and 'e' not in obj.lower():
                    return int(obj)
                else:
                    # Otherwise convert to float
                    return float(obj)
            except ValueError:
                # If conversion fails, keep as string
                return obj
        return obj
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list items
        return [convert_numeric_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively convert tuple items (return as tuple)
        return tuple(convert_numeric_strings(item) for item in obj)
    else:
        # Return as-is for other types
        return obj


def merge_with_dotlist(conf, dotlist):
    from omegaconf import OmegaConf

    def fail() -> None:
        raise ValueError("Input list must be a list or a tuple of strings")

    if not isinstance(dotlist, (list, tuple)):
        fail()

    for arg in dotlist:
        if not isinstance(arg, str):
            fail()

        idx = arg.find("=")
        if idx == -1:
            key = arg
            value = None
        else:
            key = arg[0:idx]
            value = arg[idx + 1:]
            value = yaml.unsafe_load(value)

        OmegaConf.update(conf, key, value, merge=True)
    return conf


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
    override = merge_with_dotlist(OmegaConf.create(), others)
    config = OmegaConf.merge(config, override)
    
    # Convert all numeric strings to actual numbers
    # This ensures that scientific notation like "1e-8" is properly converted
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = convert_numeric_strings(config_dict)
    config = OmegaConf.create(config_dict)
    
    override_dict = OmegaConf.to_container(override, resolve=True)
    override_dict = convert_numeric_strings(override_dict)
    override = OmegaConf.create(override_dict)

    return config, override


def get(config, mode):
    """ Create a personal dict for each executor (trainor, validator or ensemblor)
    For an executor, we give all args but the other executors args
    """
    exec_config = copy.deepcopy(getattr(config, mode))
    for att in list(config.keys()):
        if att not in ['trainor', 'validator', 'ensemblor']:
            exec_config[att] = config[att]
    return exec_config


def get_seed(seed=None):
    if seed is None:
        seed = randrange(100000, 999999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    return seed
