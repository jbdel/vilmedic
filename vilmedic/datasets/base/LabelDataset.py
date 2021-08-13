import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
from .utils import Labels
import os
import pydicom
import numpy as np
from .utils import Labels, load_file


def make_labels(root, split, file):
    return load_file(os.path.join(root, split + '.' + file))


class LabelDataset(Dataset):

    def __init__(self, root, ckpt_dir, split, file, **kwargs):

        self.root = root
        self.split = split

        self.labels = make_labels(root, split, file)
        label_file = os.path.join(ckpt_dir, 'labels')

        if split == 'train':
            labels_map = Labels(self.labels)
            if not os.path.exists(label_file): labels_map.dump(label_file)
            print('Labels:', labels_map)

        self.labels_map = Labels().load(label_file)
        self.labels = [torch.tensor(self.labels_map.label2idx(label)).long() for label in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]
