import json
import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vilmedic.executors import Trainor, Validator
from utils import get_args, get, print_args, get_seed, extract_seed_from_ckpt
from logger import set_logger
from omegaconf import OmegaConf


def main():
    # Get args and create seed
    config, override = get_args()

    seed = get_seed()

    # Create checkpoint dir
    config.ckpt_dir = os.path.join(config.ckpt_dir, config.name)
    os.makedirs(config.ckpt_dir, exist_ok=True)

    # If ckpt is specified, we continue training. Let's extract seed
    if hasattr(config, 'ckpt') and config.get('ckpt'):
        config.ckpt = os.path.join(config.ckpt_dir, config.ckpt) if not os.path.exists(config.ckpt) else config.ckpt
        assert os.path.exists(config.ckpt), f"Path '{config.ckpt}' does not exist"
        seed = extract_seed_from_ckpt(config.ckpt)

    # Create logger according to seed
    set_logger(config.ckpt_dir, seed)

    # Dump config
    json.dump(OmegaConf.to_container(config, resolve=True), open(f"{config.ckpt_dir}/config_{seed}.json", "w"),
              indent=4)

    # Nice print args
    print_args(config, ['trainor', 'validator'], seed, override)

    # Fetch args for training and validation
    train_config = get(config, 'trainor')
    val_config = get(config, 'validator')

    # Trainor
    trainor = Trainor(config=train_config,  # train_config is all args but the other executors args
                      seed=seed)

    # Evaluator
    evaluator = Validator(config=val_config,
                          models=[trainor.model],
                          train_dl=trainor.dl,
                          seed=seed,
                          from_training=True)

    # Lets be gentle, give evaluator to trainor
    trainor.evaluator = evaluator

    # Boom
    trainor.start()


if __name__ == "__main__":
    main()
