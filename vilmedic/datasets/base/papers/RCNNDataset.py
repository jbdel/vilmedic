import torch
from torch.utils.data import Dataset

from ..base.ImageDataset import ImageDataset
import os
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)


class RCNNDataset(Dataset):
    def __init__(self, label, image, bbox, split, ckpt_dir, **kwargs):
        self.split = split
        self.image = ImageDataset(**image, split=split)

        self.bbox = open(os.path.join(bbox.root, split + '.' + bbox.file)).readlines()
        self.label = open(os.path.join(label.root, split + '.' + label.file)).readlines()

        # str to list
        self.bbox = [eval(bbox) for bbox in self.bbox]
        self.label = [eval(label) for label in self.label]
        assert len(self.image) == len(self.label) == len(self.bbox)

    def __getitem__(self, index):
        bbox = self.bbox[index]
        bbox = torch.as_tensor(bbox, dtype=torch.float32)

        labels = self.label[index]
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        iscrowd = torch.zeros((len(bbox),), dtype=torch.int64)

        assert len(bbox) == len(labels)

        return {'image': self.image.__getitem__(index),
                'boxes': torch.as_tensor(bbox, dtype=torch.float32),
                'image_id': torch.tensor([index]),
                'area': area,
                'iscrowd': torch.zeros((len(bbox),), dtype=torch.int64),
                'labels': torch.as_tensor(iscrowd, dtype=torch.int64),
                }

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'images': torch.stack([s['image'] for s in batch]),
                        'targets': [{"labels": s["labels"],
                                     "boxes": s["boxes"],
                                     "image_id": s["image_id"],
                                     "iscrowd": s["iscrowd"],
                                     "area": s["area"],
                                     } for s in batch]
                        }
            return collated

        return collate_fn

    def __len__(self):
        return len(self.image)

    def __repr__(self):
        return ""
