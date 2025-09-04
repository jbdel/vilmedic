import os
from datasets import load_dataset, load_from_disk, concatenate_datasets


def load_hf_dataset(datasets, hf_local: bool, hf_filter, hf_field: str, split: str):
    """Load one or multiple HF datasets (hub or local) for a given split, apply optional filters,
    and select only the target field column for efficiency. Supports lists for datasets and hf_filter.
    """
    # Normalize to lists
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(hf_filter, str) or hf_filter is None:
        hf_filter = [hf_filter] if hf_filter is not None else []

    def process_single(name):
        load_func = load_from_disk if hf_local else load_dataset
        ds = load_func(name)
        ds = ds[split]
        for fil in hf_filter:
            ds = ds.filter(eval(fil))
        # Select only the required field
        if hf_field in ds.column_names:
            ds = ds.select_columns([hf_field])
        return ds

    if len(datasets) == 1:
        return process_single(datasets[0])
    else:
        return concatenate_datasets([process_single(name) for name in datasets])


def resolve_image_field(dataset, field, base_path=None, num_proc=8):
    def _fix(example):
        item = example[field]
        items = item if isinstance(item, list) else [item]
        fixed = []
        for img in items:
            if isinstance(img, str):
                path = os.path.join(base_path, img) if base_path is not None else img
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {img} or {path}")
                fixed.append(path)
            else:
                fixed.append(img)
        example[field] = fixed if isinstance(item, list) else [fixed[0]]
        return example

    return dataset.map(_fix, num_proc=num_proc, desc=f"Resolving {field} with base={base_path}")


