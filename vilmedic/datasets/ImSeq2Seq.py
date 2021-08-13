import torch
from torch.utils.data import Dataset
from .Seq2Seq import Seq2Seq
from .base.ImageDataset import ImageDataset


class ImSeq2Seq(Dataset):
    def __init__(self, src, tgt, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.seq2seq = Seq2Seq(src, tgt, split, ckpt_dir)
        self.image = ImageDataset(**image, split=split)

        assert len(self.image) == len(self.seq2seq)

    def __getitem__(self, index):
        return {'image': self.image.__getitem__(index), **self.seq2seq.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'images': torch.stack([s['image'] for s in batch]),
                        **self.seq2seq.get_collate_fn()(batch)}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.seq2seq)
