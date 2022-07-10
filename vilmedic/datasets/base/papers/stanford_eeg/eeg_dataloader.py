import numpy as np
import torch

import pickle

from torch.utils.data import Dataset, DataLoader
from eeg_dataloader_utils import *

from vilmedic.blocks.vision.eeg_modeling.dense_inception import DenseInception


class EegTextDataset(Dataset):
    """
    Pytorch dataset that loads both EEG clips and accompanying text report
    """

    def __init__(
        self,
        stanford_dataset_dir,
        lpch_dataset_dir,
        reports_dict_pth,
        split_type,
        clip_len=12,
    ):
        """
        Store the filenames of the seizures to use.
        Args:
            stanford_dataset_dir/lpch_dataset_dir: (string) directory containing the EEG files for stanford/lpch data
            reports_dict_pth: (string) path containing the reports dict
            split_type: (string) whether train, val, or test set
            clip_len: (int) how long EEG clip input is in seconds
        """

        with open(reports_dict_pth, "rb") as pkl_f:
            self.reports_dict = pickle.load(pkl_f)

        self.file_names = list(self.reports_dict.keys())

        eeg_file_tuples = compute_stanford_file_tuples(
            stanford_dataset_dir, lpch_dataset_dir, ["train", "dev"]
        )

        # create file_tuple dict
        eeg_file_tuple_dict = {}
        for entry in eeg_file_tuples:
            file_name = entry[0].split("/")[-1].split(".eeghdf")[0]
            # filter eeg_file_tuples to only ones with reports
            if file_name in self.file_names:
                eeg_file_tuple_dict[file_name] = entry
        self.eeg_file_tuple_dict = eeg_file_tuple_dict

        self.clip_len = clip_len

        # TODO: deal with train/val/test splits!!

    def __len__(self):
        return len(self.eeg_file_tuple_dict)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        eeg_text = self.reports_dict[file_name]

        file_path, clip_idx, split = self.eeg_file_tuple_dict[file_name]
        eeg_clip = stanford_eeg_loader(file_path, clip_idx, split, self.clip_len)

        return eeg_clip, eeg_text


class TuhEegTextDataset(Dataset):
    """
    Pytorch dataset that loads both EEG clips and accompanying text report
    """

    def __init__(
        self,
        raw_dataset_dir,
        dataset_dir,
        reports_dict_pth,
        split_type,
        clip_len=12,
    ):
        """
        Store the filenames of the seizures to use.
        Args:
            stanford_dataset_dir/lpch_dataset_dir: (string) directory containing the EEG files for stanford/lpch data
            reports_dict_pth: (string) path containing the reports dict
            split_type: (string) whether train, val, or test set
            clip_len: (int) how long EEG clip input is in seconds
        """

        # TODO: load text reports
        # retrieve paths of all edf files in the raw_dataset_dir
        # self.edf_files = []
        # for path, subdirs, files in os.walk(raw_dataset_dir):
        #     for name in files:
        #         if ".edf" in name:
        #             self.edf_files.append(os.path.join(path, name))

        self.eeg_file_tuples = compute_tuh_file_tuples(
            raw_dataset_dir, dataset_dir, split_type, clip_len, clip_len
        )
        self.dataset_dir = dataset_dir

        # # create file_tuple dict
        # eeg_file_tuple_dict = {}
        # for entry in eeg_file_tuples:
        #     file_name = entry[0].split("/")[-1].split(".edf")[0]
        #     # filter eeg_file_tuples to only ones with reports
        #     # if file_name in self.file_names:
        #     eeg_file_tuple_dict[file_name] = entry
        # self.eeg_file_tuple_dict = eeg_file_tuple_dict

        self.clip_len = clip_len

        # TODO: deal with train/val/test splits!!

    def __len__(self):
        return len(self.eeg_file_tuples)

    def __getitem__(self, idx):
        # TODO: load text reports
        # eeg_text = self.reports_dict[file_name]

        edf_fn, clip_idx, split = self.eeg_file_tuples[idx]
        h5_fn = os.path.join(self.dataset_dir, edf_fn.split(".edf")[0] + ".h5")

        eeg_clip = tuh_eeg_loader(
            h5_fn, clip_idx, clip_len=self.clip_len, stride=self.clip_len
        )

        eeg_clip = eeg_clip.swapaxes(0, 1).reshape(19, -1).T

        # return eeg_clip, eeg_text
        return eeg_clip, None


def test_dataloader_and_encoder():

    test_tuh = True

    if test_tuh:
        raw_dataset_dir = "/media/nvme_data/TUH/v1.5.2/edf"
        dataset_dir = "/media/nvme_data/siyitang/TUH_eeg_seq_v1.5.2/resampled_signal"

        eeg_ds = TuhEegTextDataset(
            raw_dataset_dir,
            dataset_dir,
            reports_dict_pth="TODO",
            split_type="train",
            clip_len=12,
        )

    else:

        stanford_dataset_dir = "/media/4tb_hdd/eeg_data/stanford/stanford_mini"
        lpch_dataset_dir = "/media/4tb_hdd/eeg_data/lpch/lpch"
        reports_dict_pth = "/media/nvme_data/eeg_reports/findings_5k_reports_dict.pkl"

        eeg_ds = EegTextDataset(
            stanford_dataset_dir,
            lpch_dataset_dir,
            reports_dict_pth,
            split_type="train",
            clip_len=12,
        )

    eeg_model = DenseInception(data_shape=(2400, 19))

    eeg_clip, eeg_text = eeg_ds[0]
    eeg_clip2, _ = eeg_ds[10]
    eeg_clip3, _ = eeg_ds[20]

    x = torch.Tensor(np.stack([eeg_clip, eeg_clip2, eeg_clip3]))

    breakpoint()

    y = eeg_model(x)

    ## for eeg encoder only, set fc2 to Identity
    eeg_encoder = DenseInception(data_shape=(2400, 19))
    eeg_encoder.fc2 = torch.nn.Identity()

    x_embs = eeg_encoder(x)

    ## use pretrained model
    pretrained_pth = "/home/ksaab/Documents/eeg_fully_supervised/results/lpch_mini2/scale_1/cv_3/best.pth.tar"
    state_dict = torch.load(pretrained_pth)["state_dict"]
    og_state_dict = eeg_model.state_dict()
    # clean up state dict keys
    state_dict_ = {}
    for key in state_dict:
        if "fc2" not in key:
            state_dict_[key.split("dense_inception.")[-1]] = state_dict[key]
        else:
            state_dict_[key.split("dense_inception.")[-1]] = og_state_dict[
                key.split("dense_inception.")[-1]
            ]

    eeg_model.load_state_dict(state_dict_)
    eeg_encoder.fc2 = torch.nn.Identity()

    x_embs = eeg_encoder(x)

    breakpoint()


if __name__ == "__main__":
    test_dataloader_and_encoder()
