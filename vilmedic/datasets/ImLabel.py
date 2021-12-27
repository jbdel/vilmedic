import torch
from torch.utils.data import Dataset

from .base.ImageDataset import ImageDataset
from .base.LabelDataset import LabelDataset


class ImLabel(Dataset):
    def __init__(self, label, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.image = ImageDataset(**image, split=split)
        self.label = LabelDataset(**label, split=split, ckpt_dir=ckpt_dir)

        assert len(self.image) == len(self.label)

    def __getitem__(self, index):
        return {**self.image.__getitem__(index), **self.label.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {**self.image.get_collate_fn()(batch),
                        **self.label.get_collate_fn()(batch)}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.image)

    def __repr__(self):
        return "ImLabel\n" + str(self.image) + '\n' + str(self.label)
