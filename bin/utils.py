import argparse
import copy
import json
from omegaconf import OmegaConf


def print_args(opts, mode):
    d = OmegaConf.to_container(opts)
    print('\033[1m\033[91m' + mode + '\033[0m')
    print(json.dumps(d, indent=4, sort_keys=True))


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
    return opts


def get(opts, mode):
    """ Create a personal dict for each executor (trainor, validator or ensemblor)
    For an executor, we give everything but the other executors args
    """
    exec_opts = copy.deepcopy(getattr(opts, mode))
    for att in list(opts.keys()):
        if att not in ['trainor', 'validator', 'ensemblor']:
            exec_opts[att] = opts[att]

    if mode == 'trainor':
        print_args(exec_opts, mode)
    else:
        print_args(getattr(opts, mode), mode)
    return exec_opts
