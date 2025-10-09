import os
from typing import Any, Dict, List

from torch.utils.data import Dataset
from PIL import Image

from ..utils_hf import load_hf_dataset, resolve_image_field
from .image_ops import get_transforms, open_image


class ImageDataset(Dataset):
    """HF-only image dataset that loads image paths/objects and applies transforms.

    Exposes:
      - images: list of list[image] (paths or PIL), always per-sample list
      - transform: callable turning PIL image to tensor
      - multi_image: optional int, used by collator in combined dataset
    """

    def __init__(self, *, split: str, image: Dict[str, Any]):
        self.split = split

        # Image config
        self.image_cfg = image
        self.image_path = image.get("image_path", None)
        self.multi_image = image.get("multi_image", 0) or 0
        self.skip_open_image = image.get("skip_open_image", False)

        hf_name = image.get("hf_dataset")
        hf_local = image.get("hf_local")
        hf_filter = image.get("hf_filter")
        field = image.get("hf_field")

        dataset = load_hf_dataset(hf_name, hf_local=hf_local, hf_filter=hf_filter, hf_field=field, split=split)
        dataset = resolve_image_field(dataset, field, base_path=self.image_path, num_proc=128)
        self.images: List[List[Any]] = [ex[field] for ex in dataset]

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
            hf_processor,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        imgs = self.images[index]
        pil_list = [open_image(p, self.skip_open_image) for p in imgs]
        tensor_list = [self.transform(im) for im in pil_list]
        return {"image": tensor_list}


