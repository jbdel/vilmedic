import os
import sys
import glob
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_args, get, print_args, get_seed
from logger import set_logger
from vilmedic.executors import Validator, create_model


def get_n_best(mode):
    n = 1
    # checking if args is formatted as best-n
    if '-' in mode:
        n = int(mode.split('-')[-1])
    return n


def get_ckpts(path, mode):
    ckpts = glob.glob(path)

    # Sort by score
    ckpts = sorted(ckpts, reverse=True)

    # Getting n-best models
    if 'best' in mode:
        n = get_n_best(mode)
        ckpts = ckpts[:n]
    return ckpts


def main():
    config, override = get_args()
    config.ckpt_dir = os.path.join(config.ckpt_dir, config.name)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    ensemble_config = get(config, 'ensemblor')
    seed = '{}_{}'.format(ensemble_config.mode, get_seed())

    set_logger(config.ckpt_dir, seed, ensemble_config.get('log_dir', None))

    # Nice printing the args
    print_args(config, ['ensemblor'], seed, override)

    # Create validator, dont give any models yet
    evaluator = Validator(config=ensemble_config,
                          models=None,
                          train_dl=None,
                          seed=seed,
                          from_training=False)

    # fetching all ckpt according to 'mode"
    ckpts = get_ckpts(os.path.join(evaluator.config.ckpt_dir, '*.pth'), ensemble_config.mode)

    # if specific checkpoint is specified
    
    if ensemble_config.get("ckpt", None) is not None:
        if os.path.isfile(ensemble_config.ckpt):
            ckpts = [ensemble_config.ckpt]
        else:
            ckpts = [os.path.join(evaluator.config.ckpt_dir, ensemble_config.ckpt)]
        assert os.path.isfile(ckpts[0]), "Specified checkpoint does not exist"

    if not ckpts:
        evaluator.logger.settings("No checkpoints found")
        sys.exit()

    evaluator.logger.settings("Checkpoints are {}".format("\n".join(ckpts)))

    # Give models to evaluator
    evaluator.models = [create_model(config=ensemble_config,
                                     # we give the first evaluation split's dataloader to model
                                     dl=evaluator.splits[0][1],
                                     logger=evaluator.logger,
                                     from_training=False,
                                     state_dict=torch.load(ckpt)).cuda().eval() for ckpt in ckpts]

    # Boom
    evaluator.start()


if __name__ == "__main__":
    main()
