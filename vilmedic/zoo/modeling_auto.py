import os
import glob
import torch
import tempfile
import transformers
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .constants import MODEL_ZOO
from .utils import download_model

from ..networks import *
from ..datasets import *

DATA_PATH = __file__.replace('vilmedic/zoo/modeling_auto.py', 'data')
transformers.logging.set_verbosity_error()


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
            file_id, _, unzip_dir = MODEL_ZOO[pretrained_model_name]
        except KeyError:
            raise KeyError("Unrecognized pretrained_model_name {}. "
                           "Model name should be one of {}.".format(pretrained_model_name, MODEL_ZOO.keys()))

        checkpoint_dir = os.path.join(DATA_PATH, unzip_dir, pretrained_model_name)

        if not os.path.exists(checkpoint_dir):
            print("Downloading in {}".format(DATA_PATH))
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
