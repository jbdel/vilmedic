import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_args, get, print_args, get_seed
from logger import set_logger
from vilmedic.executors import Trainor, Validator


def main():
    opts = get_args()
    seed = get_seed()

    set_logger(opts.ckpt_dir, seed)

    # Nice printing the args
    print_args(opts, ['trainor', 'validator'], seed)

    # Fetch args for training and validation
    train_opts = get(opts, 'trainor')
    val_opts = get(opts, 'validator')

    # Trainor
    trainor = Trainor(opts=train_opts,
                      seed=seed)

    # Evaluator
    evaluator = Validator(opts=val_opts,
                          models=[trainor.model],
                          seed=seed)

    # Lets be gentle, give evaluator to trainor
    trainor.evaluator = evaluator

    # Boom
    trainor.start()


if __name__ == "__main__":
    main()
