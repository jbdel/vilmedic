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

        self.labels_map = self.label.labels_map

    def __getitem__(self, index):
        return {**self.image.__getitem__(index), **self.label.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {**self.image.get_collate_fn()(batch),
                        **self.label.get_collate_fn()(batch)}
            return collated

        return collate_fn

    def inference(self, image=None, label=None):
        if label is None and image is None:
            return dict()

        batch = {}
        if image is not None:
            batch.update(self.image.inference(image))

        if label is not None:
            batch.update(self.label.inference(label))

        assert len(set([len(v) for k, v in batch.items()])) == 1, 'element in batch dont have the same size'
        return batch

    def __len__(self):
        return len(self.image)

    def __repr__(self):
        return "ImLabel\n" + str(self.image) + '\n' + str(self.label)
