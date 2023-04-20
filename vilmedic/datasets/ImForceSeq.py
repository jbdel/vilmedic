import torch
from torch.utils.data import Dataset
from .base.ImageDataset import ImageDataset
from .base.TextDataset import TextDataset
import json


class ImForceSeq(Dataset):
    def __init__(self, seq, force_seq, image, split, ckpt_dir, **kwargs):
        self.split = split

        self.seq = TextDataset(**seq, split=split, ckpt_dir=ckpt_dir)
        self.force_seq = TextDataset(**force_seq, split=split, ckpt_dir=ckpt_dir, separate_tokenizer_per_phrase=True)
        self.image = ImageDataset(**image, split=split)

        assert len(self.image) == len(self.seq) == len(self.force_seq)

        # For decoding, if needed
        self.tokenizer = self.seq.tokenizer
        self.tokenizer_max_len = self.seq.tokenizer_max_len

        # For tokenizing
        self.tokenizer_args = self.seq.tokenizer_args

    def __getitem__(self, index):
        return {**self.image.__getitem__(index), **self.seq.__getitem__(index), **self.force_seq.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {**self.seq.get_collate_fn()(batch), **self.image.get_collate_fn()(batch)}
            force_seq_batch = self.force_seq.get_collate_fn()(batch)
            collated['force_input_ids'] = force_seq_batch['input_ids']

            return collated


        return collate_fn

    def __len__(self):
        return len(self.image)

    def __repr__(self):
        return "ImForceSeq\n" + str(self.seq) + '\n' + str(self.force_seq) + '\n' + str(self.image)

    def inference(self, force_seq=None, seq=None, image=None):
        if seq is None and image is None:
            return dict()

        batch = {}
        if image is not None:
            batch.update(self.image.inference(image))

        if seq is not None:
            batch.update(self.seq.inference(seq))

        if force_seq is not None:
            batch.update(self.force_seq.inference(force_seq))

        assert len(set([len(v) for k, v in batch.items()])) == 1, 'elements in batch do not have the same size'
        return batch
