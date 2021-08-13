import torch
from torch.utils.data import Dataset
from .Seq2Seq import Seq2Seq
from .base.ImageDataset import ImageDataset
from .base.LabelDataset import LabelDataset
from .base.TextDataset import TextDataset


class ImSeqLabel(Dataset):
    def __init__(self, src, label, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.src = TextDataset(**src, split=split, ckpt_dir=ckpt_dir, source="src")
        self.image = ImageDataset(**image, split=split)
        self.label = LabelDataset(**label, split=split, ckpt_dir=ckpt_dir)

        assert len(self.image) == len(self.label) == len(self.src)

    def __getitem__(self, index):
        return {'image': self.image.__getitem__(index),
                'src': ' '.join(self.src.__getitem__(index)[:self.src.max_len]),
                'label': self.label.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            src = self.src.tokenizer([s['src'] for s in batch], padding=True, return_tensors="pt",
                                     add_special_tokens=False)
            collated = {'images': torch.stack([s['image'] for s in batch]),
                        'input_ids': src.input_ids,
                        'attention_mask': src.attention_mask,
                        'labels': torch.stack([s['label'] for s in batch])}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.image)
