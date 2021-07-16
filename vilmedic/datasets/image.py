import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
from .utils import Labels
import os


class ImageDataset(Dataset):

    def __init__(self, root, ckpt_dir, split, image_path, images, labels, load_memory, resize=256, **kwargs):
        assert split in ["train", "val", "test"]

        self.root = root
        self.split = split
        self.image_path = image_path
        self.load_memory = load_memory
        self.resize = resize
        self.labels = self.make_labels(root, split, labels)
        self.images = self.make_images(root, image_path, split, images)

        label_file = os.path.join(ckpt_dir, 'labels')
        if split == 'train':
            # Create vocab
            labels_map = Labels(self.labels)
            if not os.path.exists(label_file):
                labels_map.dump(label_file)
            print('Labels:', labels_map)

            self.transform = transforms.Compose([
                transforms.Resize(eval(self.resize)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            print(self.transform)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.labels_map = Labels().load(label_file)

        assert len(self.images) == len(self.labels)

        self.processed_samples = []
        for idx in range(len(self)):
            a = self.labels[idx]
            i = self.images[idx]

            label = (torch.tensor(self.labels_map.label2idx(a)).long())
            if self.load_memory:
                sample = (label, Image.open(i).convert('RGB'))
            else:
                sample = (label, i)

            self.processed_samples.append(sample)

    def make_labels(self, root, split, file):
        return self.load_file(os.path.join(root, split + '.' + file))

    def make_images(self, root, image_path, split, file):
        images = self.load_file(os.path.join(root, split + '.' + file))
        return [os.path.join(image_path, i + ".jpg") for i in images]

    def load_file(self, path):
        """Default loading function, which loads nth sentence at line n.
        """
        with open(path, 'r') as f:
            content = f.read().strip()
        return [s for s in content.split('\n')]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        a, img = self.processed_samples[index]

        if not self.load_memory:
            img = Image.open(img).convert('RGB')

        return {
            'image': self.transform(img),
            'label': a
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'image': torch.stack([s['image'] for s in batch]),
                        'label': torch.stack([s['label'] for s in batch])}
            return collated

        return collate_fn
