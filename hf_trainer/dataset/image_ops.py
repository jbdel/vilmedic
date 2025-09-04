import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_transforms(split, resize, crop, custom_transform_train, custom_transform_validate, ext, hf_processor):
    if hf_processor is not None:
        return hf_processor
    if custom_transform_train is not None and split == "train":
        return eval(custom_transform_train)
    if custom_transform_validate is not None and not split == "train":
        return eval(custom_transform_validate)
    if ext in [".npy", ".npz"]:
        return lambda x: x
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


def open_image(image, ext):
    if isinstance(image, Image.Image):
        return image if image.mode == 'RGB' else image.convert('RGB')
    if ext in ['.jpg', '.jpeg', '.png']:
        return Image.open(image).convert('RGB')
    if ext == '.dcm':
        ds = dcmread(image)
        if 'WindowWidth' in ds:
            img = apply_voi_lut(ds.pixel_array, ds).astype(float)
        else:
            img = ds.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        return Image.fromarray(img).convert('RGB')
    if ext in ['.npy', '.npz']:
        if isinstance(image, str):
            arr = np.load(image)
        else:
            arr = image
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        return arr
    raise NotImplementedError(f"Image extension {ext} not implemented")


def vilmedic_collate(batch, multi_image=None):
    if not multi_image or multi_image == 1:
        return {'images': torch.stack([s['image'][0] for s in batch]), 'images_mask': None}
    new_batch, new_masks = [], []
    for sample in batch:
        sample_images = sample['image']
        if len(sample_images) > multi_image:
            sample_images = sample_images[:multi_image]
        if len(sample_images) < multi_image:
            first_image = sample_images[0]
            for _ in range(multi_image - len(sample_images)):
                sample_images.append(first_image.new_zeros(first_image.size()))
        sample_images = torch.cat([s.unsqueeze(dim=0) for s in sample_images], dim=0)
        sample_mask = (sample_images.sum(dim=(1, 2, 3)) != 0)
        new_batch.append(sample_images)
        new_masks.append(sample_mask)
    return {'images': torch.stack(new_batch), 'images_mask': torch.stack(new_masks)}


def default_collate(batch):
    return {'images': pytorch_default_collate([s['image'][0] for s in batch]), 'images_mask': None}


# Note: Text collation is handled by TextDataset.collate in ImSeq.get_collate_fn.


