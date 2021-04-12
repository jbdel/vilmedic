import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import *
from .utils import Vocab
from .utils import Labels
import os


class VQADatasetStatic:
    vocab = None
    max_len = None
    labels = None

class VQADataset(Dataset):

    def __init__(self, root, image_path, split, questions, images, answers, load_memory, max_len,
                 **kwargs):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.image_path = image_path
        self.load_memory = load_memory

        self.questions = self.make_questions(root, split, questions)
        self.answers = self.make_labels(root, split, answers)
        self.images = self.make_images(root, image_path, split, images)

        if split == 'train':
            # Create vocab
            VQADatasetStatic.vocab = Vocab(map(lambda x: x, self.questions))
            VQADatasetStatic.labels = Labels(self.answers)
            VQADatasetStatic.max_len = max_len
            VQADatasetStatic.vocab.dump(os.path.join(root, "vocab.q"))

            print('vocab', VQADatasetStatic.vocab)
            print('labels', VQADatasetStatic.labels)
            print('train_max_len', VQADatasetStatic.max_len)

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

        self.vocab = VQADatasetStatic.vocab
        self.labels = VQADatasetStatic.labels
        self.max_len = VQADatasetStatic.max_len

        assert len(self.questions) == len(self.answers)

        self.processed_samples = []
        for idx in range(len(self)):
            q = self.questions[idx][:self.max_len] + ['[SEP]']
            a = self.answers[idx]
            i = self.images[idx]

            sample = (
                torch.tensor(self.vocab.words2idxs(q)).long(),
                torch.tensor(self.labels.label2idx(a)).long())

            if self.load_memory:
                sample = (*sample, Image.open(i).convert('RGB'))
            else:
                sample = (*sample, i)

            self.processed_samples.append(sample)

    def make_labels(self, root, split, file):
        return self.load_file(os.path.join(root, split + '.' + file))

    def make_images(self, root, image_path, split, file):
        images = self.load_file(os.path.join(root, split + '.' + file))
        return [os.path.join(image_path, i + ".jpg") for i in images]

    def make_questions(self, root, split, file):
        questions = self.load_file(os.path.join(root, split + '.' + file))
        return [q.strip().split() for q in questions]

    def load_file(self, path):
        """Default loading function, which loads nth sentence at line n.
        """
        with open(path, 'r') as f:
            content = f.read().strip()
        return [s for s in content.split('\n')]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.questions)

    def __getitem__(self, index):
        'Generates one sample of data'
        q, a, img = self.processed_samples[index]

        if not self.load_memory:
            img = Image.open(img).convert('RGB')

        return {
            'image': self.transform(img),
            'question': q,
            'label': a
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'image': torch.stack([s['image'] for s in batch]),
                        'question': pad_sequence([s['question'] for s in batch], batch_first=True),
                        'label': torch.stack([s['label'] for s in batch])}
            return collated

        return collate_fn
