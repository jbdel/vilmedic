from .image import ImageDataset
import torch
import numpy as np


class ImageInteprolateDataset(ImageDataset):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def mixup_data(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def get_collate_fn(self):
        def collate_fn(batch):
            image = torch.stack([s['image'] for s in batch])
            label = torch.stack([s['label'] for s in batch])
            mixed_images, label_true, label_mixed, lam = self.mixup_data(image, label)
            collated = {'image': mixed_images,
                        'label': label_mixed,
                        'label_mixed': label_mixed,
                        'lam': torch.FloatTensor([lam])}
            return collated

        return collate_fn
