import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import *
import os
import pydicom
import numpy as np
from .utils import load_file
import json
import PIL
import skimage
import torchxrayvision as xrv

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Are we sure ?
PIL.Image.MAX_IMAGE_PIXELS = None  # Are we sure ?


def get_transforms(split, resize, crop, custom_transform_train, custom_transform_val, ext):
    # assert both or no custom transform mentioned (xnor)
    assert not ((custom_transform_train is None) ^ (
            custom_transform_val is None)), 'Use both or no custom transform'

    assert not (custom_transform_train is not None and (ext in ['.xrv'])), \
        'Cant specify custom transform when using xrv'

    if ext == '.xrv':
        return transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                   xrv.datasets.XRayResizer(crop),
                                   lambda x: torch.from_numpy(x)])

    if split == 'train':
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]) if custom_transform_train is None else eval(
            custom_transform_train)
    else:
        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]) if custom_transform_val is None else eval(
            custom_transform_val)


def open_image(image, ext):
    if ext == '.jpg':
        return Image.open(image).convert('RGB')

    if ext == '.xrv':
        img = skimage.io.imread(image)
        img = xrv.datasets.normalize(img, 255)
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        return img[None, :, :]

    if ext == '.png':
        return Image.open(image).convert('RGB')

    if ext == '.dcm':
        ds = pydicom.dcmread(image)
        img = ds.pixel_array
        return Image.fromarray(np.uint8(img)).convert('RGB')

    if ext in ['.npy', '.npz']:
        return torch.from_numpy(np.load(image))

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
        self.file = file
        self.split = split
        self.image_path = image_path
        self.load_memory = load_memory
        self.resize = eval(str(resize))
        self.crop = eval(str(crop))
        self.ext = ext
        self.do_transform = self.ext not in [".npy", ".npz"]

        self.images = make_images(root, image_path, split, file)

        self.transform = get_transforms(split,
                                        self.resize,
                                        self.crop,
                                        custom_transform_train,
                                        custom_transform_val,
                                        ext)
        if self.load_memory:
            self.images = [open_image(image, ext) for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if not self.load_memory:
            image = open_image(image, self.ext)

        if self.do_transform:
            return self.transform(image)
        else:
            return image

    def __repr__(self):
        str_transforms = None
        if self.do_transform:
            str_transforms = self.transform
            if hasattr(self.transform, "transform"):
                str_transforms = self.transform.transforms

        return "ImageDataset\n" + \
               json.dumps({
                   "image_path": self.image_path,
                   "root": self.root,
                   "file": self.file,
                   "transforms": str_transforms,
                   "ext": self.ext,
                   "do_transform": self.do_transform,
               }, indent=4, sort_keys=False, default=str)
