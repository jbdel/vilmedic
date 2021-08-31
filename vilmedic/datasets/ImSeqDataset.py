import torch
from torch.utils.data import Dataset
from .base.ImageDataset import ImageDataset
from .base.TextDataset import TextDataset


class ImSeqDataset(Dataset):
    def __init__(self, seq, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.seq = TextDataset(**seq, split=split, ckpt_dir=ckpt_dir)
        self.image = ImageDataset(**image, split=split)

        assert len(self.image) == len(self.seq)

        # For decoding
        self.tokenizer = self.seq.tokenizer
        self.max_len = self.seq.max_len

        # For tokenizer
        self.tokenizer_max_len = self.seq.tokenizer_max_len or self.seq.max_len

    def __getitem__(self, index):
        return {'image': self.image.__getitem__(index),
                'seq': ' '.join(self.seq.__getitem__(index)[:self.max_len])
                if (self.split == 'train' and self.seq.source == "tgt")
                else ' '.join(self.seq.__getitem__(index))  # No trunc at test time
                }

    def get_collate_fn(self):
        def collate_fn(batch):
            seq = self.tokenizer([s['seq'] for s in batch], padding='max_length', truncation=True,
                                 max_length=self.tokenizer_max_len, return_tensors="pt")
            collated = {'images': torch.stack([s['image'] for s in batch]),
                        'input_ids': seq.input_ids,
                        'attention_mask': seq.attention_mask
                        }
            return collated

        return collate_fn

    def __len__(self):
        return len(self.image)
