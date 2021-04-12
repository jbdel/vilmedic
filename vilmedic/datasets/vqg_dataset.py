import torch
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import *
from .text_rnn import TextDatasetRNN


class VQGDataset(TextDatasetRNN):

    def __init__(self, images, image_path, load_memory, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.load_memory = load_memory
        self.images = self.make_images(self.root, image_path, self.split, images)

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        assert len(self) == super().__len__()

        self.processed_images = []
        for idx in range(len(self)):
            i = self.images[idx]
            if self.load_memory:
                sample = Image.open(i).convert('RGB')
            else:
                sample = i
            self.processed_images.append(sample)

    def make_images(self, root, image_path, split, file):
        images = self.load_file(os.path.join(root, split + '.' + file))
        return [os.path.join(image_path, i + ".jpg") for i in images]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        text = super().__getitem__(index)
        img = self.processed_images[index]
        if not self.load_memory:
            img = Image.open(img).convert('RGB')

        return dict({'image': self.transform(img)},
                    **text)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'src': torch.stack([s['image'] for s in batch]),
                        # 'src': pad_sequence([s['src'] for s in batch], batch_first=False), # dont use
                        'tgt': pad_sequence([s['tgt'] for s in batch], batch_first=False)
                        }
            return collated

        return collate_fn
