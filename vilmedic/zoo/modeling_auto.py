import os
import glob
import torch
import transformers
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from vilmedic.constants import MODEL_ZOO_CACHE_DIR
from .utils import download_model

from ..networks import *
from ..datasets import *

transformers.logging.set_verbosity_error()

MODEL_ZOO = {
    'selfsup/gloria-chexpert': ["1ibtEQH8XXNPy1Y0fE1ooSF7Jh7bdG58C", "1.6 GB"],
    'selfsup/gloria-mimic-48': ["1V50_NUWY-k2ebzmgQxRUcMOIH0UhSAEr", "1.6 GB"],
    'selfsup/convirt-mimic-balanced': ["1bTf16uRygWwTor3X0rYBhD7MuHSUjWxM", "1.4 GB"],
    'selfsup/convirt-mimic': ["1py2k6kFO1tbSlHroAKmHjOnayIakOtMm", "1.4 GB"],
    'selfsup/convirt-padchest-16': ["15p6ZaoqcxAB0dct7P9zgOcTfvZw-XkVv", "1.4 GB"],
    'selfsup/convirt-padchest-32': ["1sd9eNVDcZTPrpmlHSqjMP6hRBHyM7e4p", "1.4 GB"],
    'selfsup/convirt-indiana-16': ["10k9RlLJVLH1tuuSjzwHQK1BTmTVOUinH", "1.4 GB"],
    'selfsup/convirt-indiana-32': ["17q0MllKOnGQY7zudhK03sqDsrnf_THu2", "1.4 GB"],
    'selfsup/convirt-indiana-64': ["17xi8Mj3Ts9qeFT0l83a5Hm82W1ocoYCW", "1.4 GB"],
    'selfsup/simclr-mimic-32': ["1ibtEQH8XXNPy1Y0fE1ooSF7Jh7bdG58C", "300 MB"],
    'selfsup/simclr-mimic-64': ["1RYhQkaR9F0LbozVs7hHv0c52Js1LDh6J", "300 MB"],
    'selfsup/simclr-mimic-128': ["1w1XYaprrJrjIk-JlKpbw7OSe3sABKDkN", "300 MB"],
    'rrg/biomed-roberta-baseline-mimic': ["1aXxHkzbLdYQpLYvlQLw7NENE7LXgkc1y", "1.8 GB"],
    'rrg/biomed-roberta-baseline-indiana': ["1BzTPf4AMLF_2KGs6RX3W30HyekeUElmW", "1.8 GB"],
    'rrg/baseline-padchest': ["1COYPFZJTiG5TBlhGSX7GyswXwKL6HAW0", "1.8 GB"],
    'rrs/biomed-roberta-baseline-mimic': ["1hmEvUjKOlNsY-xipEgUZOCQm4k9mHgWR", "3.3 GB"],
    'rrs/biomed-roberta-baseline-indiana': ["1xG80gsckbdNvAVhqGo-4Lsvkwk7wy_-v", "3.3 GB"],
    'mvqa/mvqa-imageclef': ["1VmiJEGs-jYNGlbVXGi6uGmdhc06Ps4GF", "970 MB"],
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModel.from_config(config)` methods."
        )

    @staticmethod
    def from_config(config):
        return None

    @staticmethod
    def from_pretrained(pretrained_model_name):
        try:
            file_id, _ = MODEL_ZOO[pretrained_model_name]
        except KeyError:
            raise KeyError("Unrecognized pretrained_model_name {}. "
                           "Model name should be one of {}.".format(pretrained_model_name, MODEL_ZOO.keys()))

        checkpoint_dir = os.path.join(MODEL_ZOO_CACHE_DIR, pretrained_model_name)

        if not os.path.exists(checkpoint_dir):
            print("Downloading in {}".format(MODEL_ZOO_CACHE_DIR))
            download_model(file_id=file_id, unzip_dir=checkpoint_dir)

        checkpoint = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        assert len(checkpoint) == 1, "More than one or no checkpoint found"
        state_dict = torch.load(checkpoint[0])

        try:
            config = OmegaConf.load(os.path.join(checkpoint_dir, 'config.yml'))
        except FileNotFoundError:
            raise FileNotFoundError("The file config.yml is missing")

        try:
            model_config = config["model"]
            dataset_config = config["dataset"]
        except KeyError:
            raise KeyError(
                "This config doesnt have a model and/or dataset key. Deprecated checkpoint of vilmedic version?")

        try:
            classname = dataset_config.pop("proto")
            dataset = eval(classname)(split='test', ckpt_dir=None, **dataset_config)
        except NameError:
            raise NameError(
                "Dataset {} does not exist anymore. Deprecated checkpoint of vilmedic version?".format(classname))

        try:
            classname = model_config.pop("proto")
            model: nn.Module = eval(classname)(**model_config, dl=DataLoader(dataset), logger=None)
        except NameError:
            raise NameError(
                "Model {} does not exists anymore. Deprecated checkpoint of vilmedic version?".format(classname))

        try:
            model.load_state_dict(state_dict["model"], strict=True)
        except Exception as e:
            print(e)
            raise

        model.eval()
        model.cuda()

        assert hasattr(dataset, "inference"), "Dataset has not implemented an inference function"

        print("Everything has been loaded successfully")

        return model, dataset
