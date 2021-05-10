import torch
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import *
from .text_hug import TextDatasetHug


def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return [s for s in content.split('\n')]


class VQGDatasetHug(TextDatasetHug):

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
        images = load_file(os.path.join(root, split + '.' + file))
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
            tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, return_tensors="pt")
            # print(self.tgt_tokenizer.decode(tgt.input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
            # troll

            collated = {'input_ids': torch.stack([s['image'] for s in batch]),
                        'decoder_input_ids': tgt.input_ids,
                        'decoder_attention_mask': tgt.attention_mask}
            return collated

        return collate_fn
