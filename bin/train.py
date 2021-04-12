import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_args, get, print_args
from vilmedic.executors import Trainor, Validator


def main():
    opts = get_args()

    train_opts = get(opts, 'trainor')
    val_opts = get(opts, 'validator')

    # Trainor
    trainor = Trainor(opts=train_opts)

    # Evaluator
    evaluator = Validator(opts=val_opts,
                          models=[trainor.model],
                          seed=trainor.seed)

    # Lets be gentle, give evaluator to trainor
    trainor.evaluator = evaluator

    # Boom
    trainor.start()


if __name__ == "__main__":
    main()
