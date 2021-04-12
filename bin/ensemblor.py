import os
import sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_args, get
from vilmedic.executors import Validator, create_model, create_data_loader


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
    opts = get_args()
    ensemble_opts = get(opts, 'ensemblor')
    evaluator = Validator(opts=ensemble_opts,
                          models=None,
                          seed='{}_{}'.format(ensemble_opts.mode, ensemble_opts.beam_width))

    ckpts = get_ckpts(os.path.join(evaluator.ckpt_dir, '*.pth'), ensemble_opts.mode)
    evaluator.models = [create_model(opts=ensemble_opts, state_dict=ckpt).cuda().eval() for ckpt in ckpts]

    # Boom
    evaluator.start()


if __name__ == "__main__":
    main()
