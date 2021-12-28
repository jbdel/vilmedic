import torch
from torch.utils.data import Dataset
from .base.ImageDataset import ImageDataset
from .base.TextDataset import TextDataset
import json


class ImSeq(Dataset):
    def __init__(self, seq, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.seq = TextDataset(**seq, split=split, ckpt_dir=ckpt_dir)
        self.image = ImageDataset(**image, split=split)

        assert len(self.image) == len(self.seq)

        # For decoding, if needed
        self.tokenizer = self.seq.tokenizer
        self.tokenizer_max_len = self.seq.tokenizer_max_len

        # For tokenizing
        self.tokenizer_args = self.seq.tokenizer_args

    def __getitem__(self, index):
        return {**self.image.__getitem__(index), **self.seq.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {**self.seq.get_collate_fn()(batch), **self.image.get_collate_fn()(batch)}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.image)

    def __repr__(self):
        return "ImSeq\n" + str(self.seq) + '\n' + str(self.image)

    def inference(self, seq=None, image=None):
        if seq is None and image is None:
            return dict()

        batch = {}
        if image is not None:
            batch.update(self.image.inference(image))

        if seq is not None:
            batch.update(self.seq.inference(seq))

        assert len(set([len(v) for k, v in batch.items()])) == 1, 'elements in batch do not have the same size'
        return batch
