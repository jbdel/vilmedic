import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
from .utils import Labels
import os
import pydicom
import numpy as np
from .utils import Labels, load_file
import json


def make_labels(root, split, file):
    return load_file(os.path.join(root, split + '.' + file))


class LabelDataset(Dataset):

    def __init__(self, root, ckpt_dir, split, file, **kwargs):
        self.root = root
        self.split = split

        self.labels = make_labels(root, split, file)
        label_file = os.path.join(ckpt_dir, 'labels')

        if split == 'train' and not os.path.exists(label_file):
            Labels(self.labels).dump(label_file)

        self.labels_map = Labels().load(label_file)

        labels = []
        for label in self.labels:
            try:
                labels.append(torch.tensor(self.labels_map.label2idx[label]).long())
            except KeyError:
                # Label in this split is not present in train set, this can happen
                labels.append(torch.tensor(-100).long())
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]

    def __repr__(self):
        return "LabelDataset\n" + \
               json.dumps({"num_labels": len(self.labels_map.labels)}, indent=4,
                          sort_keys=False, default=str)
