import copy
from typing import Any, Dict

from torch.utils.data import Dataset

from .image.dataset import ImageDataset
from .text.dataset import TextDataset
from .image.image_ops import vilmedic_collate, default_collate


class VisionLanguageDataset(Dataset):
    """Wrapper that combines an image dataset and a text dataset and owns the collate logic."""

    def __init__(self, *, seq: Dict[str, Any], image: Dict[str, Any], split: str, ckpt_dir: str,  max_samples: int=None, **kwargs):
        self.split = split
        self.image_ds = ImageDataset(split=split, image=image)
        self.text_ds = TextDataset(split=split, ckpt_dir=ckpt_dir, **seq)
        
        assert len(self.image_ds) == len(self.text_ds)

        # Expose tokenizer for trainer and metrics
        self.tokenizer = self.text_ds.tokenizer
        self.tokenizer_max_len = self.text_ds.tokenizer_max_len
        self.tokenizer_args = self.text_ds.tokenizer_args

        # For collate
        self.multi_image = self.image_ds.multi_image

        # Misc
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples:
            return min(len(self.image_ds), self.max_samples)
        return len(self.image_ds)

    def __getitem__(self, index):
        # Build independently and merge keys; text collator will re-tokenize from strings
        sample = {}
        sample.update(self.image_ds[index])
        sample.update(self.text_ds[index])
        return sample

    def get_collate_fn(self):
        text_collate = self.text_ds.get_collate_fn()

        def collate_fn(batch):
            # 1) build vision tensors (+ masks)
            vision = (
                vilmedic_collate(batch, self.multi_image)
                if self.multi_image and self.multi_image > 1
                else default_collate(batch)
            )
            # 2) build text tensors via text dataset collator
            text = text_collate(batch)
            # 3) merge
            vision.update(text)
            return vision

        return collate_fn

    def __repr__(self):
        return (
            "VisionLanguageDataset(HF-only)\n"
            f"split={self.split}, len={len(self)}, multi_image={self.multi_image}\n"
            f"tokenizer={self.tokenizer.name_or_path}, max_len={self.tokenizer_max_len}"
        )


def create_dataset(config, split, logger):
    dataset_config = copy.deepcopy(config.dataset)
    proto = dataset_config.get('proto')
    # Maintain compatibility: accept 'ImSeq' and new name 'VisionLanguage'
    assert proto in ["ImSeq", "VisionLanguage"], f"Only VisionLanguage/ImSeq dataset is supported, got {proto}"

    dataset_kwargs = {k: v for k, v in dataset_config.items() if k != 'proto'}
    dataset = VisionLanguageDataset(
        split=split,
        ckpt_dir=config.ckpt_dir,
        **dataset_kwargs
    )

    # Log dataset information
    try:
        multi_image = getattr(dataset, 'multi_image', None)
        logger.info(f"[Dataset] {split.capitalize()} dataset loaded:")
        logger.info(f"  Size: {len(dataset):,} samples")
        if multi_image:
            logger.info(f"  Multi-image mode: {multi_image} images per sample")
        logger.info(f"  Tokenizer vocab size: {dataset.tokenizer.vocab_size:,}")
    except Exception:
        logger.info(f"[Dataset] {split.capitalize()} dataset loaded: {dataset}")

    return dataset


__all__ = ["VisionLanguageDataset", "create_dataset"]


