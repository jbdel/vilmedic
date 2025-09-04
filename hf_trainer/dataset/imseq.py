import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils_hf import load_hf_dataset, resolve_image_field
from .image_ops import get_transforms, open_image, vilmedic_collate, default_collate
from .text_dataset import TextDataset


class ImSeq(Dataset):
    def __init__(self, seq, image, split, ckpt_dir, **kwargs):
        self.split = split

        # Text side (HF-only)
        self.seq = TextDataset(**seq, split=split, ckpt_dir=ckpt_dir)

        # Image side (HF-only)
        self.image_cfg = image
        self.image_path = image.get("image_path", None)
        self.multi_image = image.get("multi_image", 0) or 0
        self.ext = image.get("ext", ".jpg")

        hf_name = image.get("hf_dataset")
        hf_local = image.get("hf_local")
        hf_filter = image.get("hf_filter")
        field = image.get("hf_field")

        dataset = load_hf_dataset(hf_name, hf_local=hf_local, hf_filter=hf_filter, hf_field=field, split=split)
        dataset = resolve_image_field(dataset, field, base_path=self.image_path, num_proc=8)
        self.images = [ex[field] for ex in dataset]

        # Transforms identical to original defaults
        resize = int(image.get("resize", 256))
        crop = int(image.get("crop", 224))
        custom_transform_train = image.get("custom_transform_train")
        custom_transform_validate = image.get("custom_transform_validate")
        hf_processor = image.get("hf_processor")
        self.transform = get_transforms(
            split,
            resize,
            crop,
            custom_transform_train,
            custom_transform_validate,
            self.ext,
            hf_processor,
        )

        # For decoding convenience
        self.tokenizer = self.seq.tokenizer
        self.tokenizer_max_len = self.seq.tokenizer_max_len
        self.tokenizer_args = self.seq.tokenizer_args

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgs = self.images[index]
        pil_list = [open_image(p, self.ext) for p in imgs]
        tensor_list = [self.transform(im) for im in pil_list]
        text = self.seq.__getitem__(index)
        # merge image list with text fields
        out = {"image": tensor_list}
        out.update(text)
        return out

    def get_collate_fn(self):
        def collate_fn(batch):
            # 1) build vision tensors (+ masks)
            vision = (
                vilmedic_collate(batch, self.multi_image)
                if self.multi_image and self.multi_image > 1
                else default_collate(batch)
            )
            # 2) build text tensors via text dataset collator
            text = self.seq.get_collate_fn()(batch)
            # 3) merge
            vision.update(text)
            return vision
        return collate_fn

    def __repr__(self):
        return (
            "ImSeq(HF-only)\n"
            f"split={self.split}, len={len(self)}, multi_image={self.multi_image}, ext={self.ext}\n"
            f"tokenizer={self.tokenizer.name_or_path}, max_len={self.tokenizer_max_len}"
        )


