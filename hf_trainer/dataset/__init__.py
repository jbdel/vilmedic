import copy

from .imseq import ImSeq


def create_dataset(config, split, logger):
    dataset_config = copy.deepcopy(config.dataset)
    proto = dataset_config.get('proto')
    assert proto == "ImSeq", f"Only ImSeq dataset is supported, got {proto}"

    dataset_kwargs = {k: v for k, v in dataset_config.items() if k != 'proto'}
    dataset = ImSeq(
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
    except Exception as e:
        logger.info(f"[Dataset] {split.capitalize()} dataset loaded: {dataset}")

    return dataset

__all__ = ["ImSeq", "create_dataset"]


