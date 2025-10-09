import os
from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
from .custom_image_ops import maira2_tf

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
CLASSIC_EXTS = set([".jpg", ".jpeg", ".png"])
BINARY_EXTS = set([".jph", ".jp2"])

custom_transforms_factory = {
    "maira2_tf": maira2_tf,
}


def _dcm_to_pil_rgb(path: str) -> Image.Image:
    ds = dcmread(path)

    # 1) Modality LUT (handles Rescale Slope/Intercept)
    img = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)

    # 2) VOI LUT / Windowing
    # apply_voi_lut works whether it’s VOI LUT table or WindowCenter/Width
    img = apply_voi_lut(img, ds).astype(np.float32)

    # 3) Normalize to [0,255]
    vmin, vmax = float(np.min(img)), float(np.max(img))
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin) * 255.0
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = img.astype(np.uint8)

    # 4) Handle MONOCHROME1 (invert)
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        img = 255 - img

    return Image.fromarray(img).convert("RGB")

def load_jp2_and_to_rgb_u8(path: str):
    # 1) Avoid extra copy
    jp2 = Image.open(path)
    # 2) Get a 16-bit array once
    arr16 = np.asarray(jp2, dtype=np.uint16)
    # 3) Normalize to uint8 (vectorized)
    vmin = arr16.min()
    vmax = arr16.max()
    if vmax <= vmin:
        arr8 = np.zeros_like(arr16, dtype=np.uint8)
    else:
        arr8 = ((arr16 - vmin) * (255.0 / (vmax - vmin))).astype(np.uint8)
    # 4) Make PIL image without round-tripping formats multiple times
    return Image.fromarray(arr8, mode="L").convert("RGB")

def get_transforms(split, resize, crop, custom_transform_train, custom_transform_validate, hf_processor):
    if hf_processor is not None:
        return hf_processor
    if custom_transform_train is not None and split == "train":
        return custom_transforms_factory[custom_transform_train]
    if custom_transform_validate is not None and not split == "train":
        return custom_transforms_factory[custom_transform_validate]
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


def open_image(image, skip_open_image):
    if skip_open_image:
        return image
        
    if isinstance(image, Image.Image):
        return image if image.mode == 'RGB' else image.convert('RGB')
        
    if isinstance(image, str):
        if os.path.splitext(image)[1].lower() in CLASSIC_EXTS:
            return Image.open(image).convert('RGB')

        if os.path.splitext(image)[1].lower() in BINARY_EXTS:
            return load_jp2_and_to_rgb_u8(image)

        if os.path.splitext(image)[1].lower() == '.dcm':
            return _dcm_to_pil_rgb(image)
        raise NotImplementedError(f"open_image not implemented for {image}")
    else:
        raise NotImplementedError(f"open_image not implemented for {image}")


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