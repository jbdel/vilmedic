import os
import pydicom
import numpy as np
import json
import PIL
import skimage
import logging

from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
import torchxrayvision as xrv
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import *
from .utils import load_file
from .papers.open_image import *
from .papers.transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Are we sure ?
PIL.Image.MAX_IMAGE_PIXELS = None  # Are we sure ?
logging.getLogger('PIL').setLevel(logging.WARNING)


def vilmedic_collate(batch, multi_image=None):
    # vilmedic_collate only accepts list of tensors (list of images that are transformed)

    # Return one image
    if not multi_image or multi_image == 1:
        return {'images': torch.stack([s['image'][0] for s in batch])}

    # Return multiple image
    new_batch = []
    new_masks = []
    for sample in batch:
        sample_images = sample['image']
        # Remove image to get to multi_image
        if len(sample_images) > multi_image:
            sample_images = sample_images[:multi_image]
        # Pad with zeros to get to multi_image
        if len(sample_images) < multi_image:
            first_image = sample_images[0]
            for _ in range(multi_image - len(sample_images)):
                sample_images.append(first_image.new_zeros(first_image.size()))
        # Stack
        sample_images = torch.cat([s.unsqueeze(dim=0) for s in sample_images], dim=0)
        sample_mask = (sample_images.sum(dim=(1, 2, 3)) != 0)
        new_batch.append(sample_images)
        new_masks.append(sample_mask)

    collated = {'images': torch.stack(new_batch),
                'images_mask': torch.stack(new_masks)}
    return collated


def default_collate(batch):
    collated = {'images': pytorch_default_collate([s['image'][0] for s in batch]),
                'images_mask': None}
    return collated


def read_images(root, image_path, split, file):
    lines = load_file(os.path.join(root, split + '.' + file))
    return [[os.path.join(image_path, image) for image in line.split(',')] for line in lines]


def get_transforms(split, resize, crop, custom_transform_train, custom_transform_val, ext, called_by_ensemblor):
    # If called_by_ensemblor, return custom_transform_val or evaluation transform
    if called_by_ensemblor:
        split = not "train"

    if custom_transform_train is not None and split == "train":
        return eval(custom_transform_train)

    if custom_transform_val is not None and not split == "train":
        return eval(custom_transform_val)

    if ext in [".npy", ".npz"]:
        return lambda x: x

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
                                 (0.229, 0.224, 0.225))])
    else:
        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])


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

    # Special cases
    if ext in PAPER_EXT.keys():
        return eval(PAPER_EXT[ext])(image)

    raise NotImplementedError("Image extension {} not implemented".format(ext))


def do_image(image, transform, ext):
    opened_images = [open_image(im, ext) for im in image]
    transformed_images = [transform(im) for im in opened_images]
    return transformed_images


class ImageDataset(Dataset):
    def __init__(self,
                 root=None,
                 file=None,
                 split=None,
                 image_path=None,
                 load_memory=False,
                 custom_transform_train=None,
                 custom_transform_val=None,
                 resize=256,
                 crop=224,
                 ext='.jpg',
                 multi_image=None,
                 called_by_ensemblor=None,
                 **kwargs):

        assert split is not None, "Argument split cant be None"

        self.root = root
        self.file = file
        self.split = split
        self.image_path = image_path
        self.load_memory = load_memory
        self.resize = eval(str(resize))
        self.crop = eval(str(crop))
        self.ext = ext
        self.multi_image = multi_image or 0
        self.images = None

        if file is not None:
            self.images = read_images(root, image_path, split, file)

        self.transform = get_transforms(split,
                                        self.resize,
                                        self.crop,
                                        custom_transform_train,
                                        custom_transform_val,
                                        self.ext,
                                        called_by_ensemblor)

    def __len__(self):
        return len(self.images or [])

    def __getitem__(self, index):
        return {'image': do_image(self.images[index], self.transform, self.ext)}

    def inference(self, image):
        if not isinstance(image, list):
            image = [image]
        batch = [{'image': do_image(i, self.transform, self.ext)} for i in image]
        return self.get_collate_fn()(batch)

    def get_collate_fn(self):
        def collate_fn(batch):
            try:
                return vilmedic_collate(batch, self.multi_image)
            except TypeError:
                return default_collate(batch)

        return collate_fn

    def __repr__(self):
        transform = self.transform
        if hasattr(self.transform, "transforms"):
            transform = self.transform.transforms

        return "ImageDataset\n" + \
               json.dumps({
                   "image_path": self.image_path,
                   "root": self.root,
                   "file": self.file,
                   "transforms": transform,
                   "ext": self.ext,
               }, indent=4, sort_keys=False, default=str)
