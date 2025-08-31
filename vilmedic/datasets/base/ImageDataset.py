import torch
import os
import pydicom
import json
import PIL
import tqdm
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import *
from transformers.image_utils import PILImageResampling

from .utils import load_file, process_hf_dataset
from .papers.transforms import *
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
from transformers import ViTImageProcessor
import multiprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Are we sure ?
PIL.Image.MAX_IMAGE_PIXELS = None  # Are we sure ?


def vilmedic_collate(batch, multi_image=None):
    # vilmedic_collate only accepts list of tensors (list of images that are transformed)

    # Return one image
    if not multi_image or multi_image == 1:
        return {'images': torch.stack([s['image'][0] for s in batch]),
                'images_mask': None}

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


def read_images(root, split, file):
    file_path = os.path.join(root, split + '.' + file)
    if '.npy' in file_path:
        return [[x] for x in np.load(file_path)]

    lines = load_file(file_path)
    images = []
    for line in lines:
        paths = line.split(',')
        # Assert each path exists
        for p in paths:
            img_path = p.strip()
            assert os.path.exists(img_path), f"Image path does not exist: {img_path}"
        images.append([p.strip() for p in paths])

    return images

def get_transforms(split, resize, crop, custom_transform_train, custom_transform_validate, ext, hf_processor, called_by_ensemblor):
    # If called_by_ensemblor, return custom_transform_validate or evaluation transform
    if called_by_ensemblor:
        split = not "train"

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
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    else:
        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])


def open_image(image, ext):
    if isinstance(image, Image.Image):
        if image.mode == 'RGB':
            return image
        else:
            return image.convert('RGB')

    if ext == '.jpg' or ext == '.jpeg':
        return Image.open(image).convert('RGB')

    if ext == '.png':
        return Image.open(image).convert('RGB')

    if ext == '.dcm':
        ds = pydicom.dcmread(image)
        if 'WindowWidth' in ds:
            img = apply_voi_lut(ds.pixel_array, ds).astype(float)
        else:
            img = ds.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        return Image.fromarray(img).convert('RGB')

    if ext in ['.npy', '.npz']:
        if type(image) == str:
            image = np.load(image)
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)
        return image


    raise NotImplementedError("Image extension {} not implemented".format(ext))


def do_images(images, transform, ext):
    opened_images = [open_image(im, ext) for im in images]
    transformed_images = [transform(im) for im in opened_images]
    return transformed_images


class ImageDataset(Dataset):
    def __init__(self,
                 root=None,
                 file=None,
                 split=None,
                 image_path=None,
                 custom_transform_train=None,
                 custom_transform_validate=None,
                 resize=256,
                 crop=224,
                 ext='.jpg',
                 multi_image=None,
                 called_by_ensemblor=None,
                 hf_dataset=None,
                 hf_field=None,
                 hf_local=None,
                 hf_filter=None,
                 hf_processor=None,
                 **kwargs):

        assert split is not None, "Argument split cant be None"

        assert hf_dataset is None or (hf_field is not None), "If 'hf_dataset' is not None, " \
                                                             "then 'hf_field' must also " \
                                                             "be not None."
        # You cannot have file and hf_dataset at the same time
        assert file is None or hf_dataset is None, "You cannot have file and hf_dataset at the same time"

        self.root = root
        self.file = file
        self.split = split
        self.image_path = image_path
        self.resize = eval(str(resize))
        self.crop = eval(str(crop))
        self.ext = ext
        self.multi_image = multi_image or 0
        self.images = None

        if file is not None:
            self.images = read_images(root, split, file)

        if hf_dataset is not None:
            dataset = process_hf_dataset(hf_dataset, hf_local, hf_filter, hf_field, split)
            
            # Check if data is list or single items
            first_example = dataset[0][hf_field]
            is_list = isinstance(first_example, list)
            
            # Validate and fix paths using HF dataset's map (multithreaded)
            def validate_and_fix_paths(example):
                """Validate image paths and fix with image_path if needed"""
                item = example[hf_field]
                
                # Handle both single items and lists
                items = item if isinstance(item, list) else [item]
                validated_items = []
                
                for img in items:
                    if isinstance(img, str):
                        if self.image_path:
                            full_path = os.path.join(self.image_path, img)
                            if os.path.exists(full_path):
                                validated_items.append(full_path)
                            else:
                                raise FileNotFoundError(f"Image file not found: {img} or {full_path}")
                        else:
                            raise FileNotFoundError(f"Image file not found: {img}")
          
                    elif isinstance(img, Image.Image):
                        validated_items.append(img)
                    else:
                        raise TypeError(f"Unexpected image type: {type(img)}")
                
                # Return in same format as input
                example[hf_field] = validated_items if isinstance(item, list) else validated_items[0]
                return example
            
            # Apply validation using map (multithreaded)
            cpu_count = multiprocessing.cpu_count()
            dataset = dataset.map(
                validate_and_fix_paths,
                desc="Validating image paths (cpu count: {})".format(cpu_count),
                num_proc=cpu_count,
            )
            
            # Convert to list format expected by rest of code
            if not is_list:
                self.images = [[d[hf_field]] for d in dataset]
            else:
                self.images = [d[hf_field] for d in dataset]


        self.transform = get_transforms(split,
                                        self.resize,
                                        self.crop,
                                        custom_transform_train,
                                        custom_transform_validate,
                                        self.ext,
                                        hf_processor,
                                        called_by_ensemblor)

    def __len__(self):
        return len(self.images or [])

    def __getitem__(self, index):
        return {'image': do_images(self.images[index], self.transform, self.ext)}

    def get_collate_fn(self):
        def collate_fn(batch):
            try:
                return vilmedic_collate(batch, self.multi_image)
            except TypeError:
                return default_collate(batch)

        return collate_fn

    def to_huggingface_processor(self):
        try:
            return ViTImageProcessor(
                **{
                    "do_normalize": True,
                    "do_resize": True,
                    "do_rescale": True,
                    "image_mean": list(self.transform.transforms[-1].mean),
                    "image_std": list(self.transform.transforms[-1].std),
                    "resample": PILImageResampling.BILINEAR,
                    "size": self.transform.transforms[0].size,
                }
            )
        except Exception as e:
            print("custom transforms has been provided and is not compatible with this method.")
            print(e)
            return None

    def __repr__(self):
        transform = self.transform
        if hasattr(self.transform, "transforms"):
            transform = self.transform.transforms

        return "ImageDataset\n" + \
            json.dumps({
                "split": self.split,
                "len": len(self),
                "image_path": self.image_path,
                "root": self.root,
                "file": self.file,
                "transforms": transform,
                "ext": self.ext,
            }, indent=4, sort_keys=False, default=str)
