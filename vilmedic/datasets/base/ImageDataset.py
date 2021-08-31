import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
import os
import pydicom
import numpy as np
from .utils import load_file


def open_image(image, ext):
    if ext == '.jpg':
        return Image.open(image)
    elif ext == '.dcm':
        ds = pydicom.dcmread(image)
        img = ds.pixel_array
        return Image.fromarray(np.uint8(img))
    else:
        raise NotImplementedError("Image extension {} not implemented".format(ext))


def make_images(root, image_path, split, file):
    images = load_file(os.path.join(root, split + '.' + file))
    return [os.path.join(image_path, image) for image in images]


class ImageDataset(Dataset):
    def __init__(self, root, image_path, file, split, load_memory,
                 resize=256,
                 crop=224,
                 custom_transform_train=None,
                 custom_transform_val=None,
                 ext='.jpg',
                 **kwargs):

        self.root = root
        self.split = split
        self.image_path = image_path
        self.load_memory = load_memory
        self.resize = eval(str(resize))
        self.crop = eval(str(crop))
        self.ext = ext

        self.images = make_images(root, image_path, split, file)

        # assert both or no custom transform mentioned (xnor)
        assert not ((custom_transform_train is None) ^ (
                custom_transform_val is None)), 'Use both or no custom transform'

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.RandomCrop(self.crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))]) if custom_transform_train is None else eval(
                custom_transform_train)

            print(self.transform)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.crop, self.crop)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))]) if custom_transform_val is None else eval(
                custom_transform_val)

        if self.load_memory:
            self.images = [open_image(image, ext).convert('RGB') for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if not self.load_memory:
            image = open_image(image, self.ext).convert('RGB')
        return self.transform(image)
