import itertools
from datasets import load_dataset, load_from_disk
import tqdm
from PIL import Image
from typing import Union, List


def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return [s for s in content.split('\n')]


class Vocab:
    def __init__(self, sentences=None,
                 pad_token="[PAD]",
                 eos_token="[SEP]",
                 bos_token="[CLS]",
                 unk_token="[UNK]",
                 mask_token="[MASK]"):
        self.vocab = list(itertools.chain(*sentences))
        self.words = [bos_token, pad_token, eos_token, unk_token, mask_token] + sorted(set(self.vocab))

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.words))


class Labels:
    def __init__(self, labels=None):
        if labels is not None:
            self.labels = list(set([l for label in labels for l in label.split(',')]))
            self.multi_label = max([len(label.split(',')) for label in labels]) > 1

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in ["multi-label:" + str(self.multi_label)] + self.labels))

    def load(self, path):
        self.labels = [w.strip() for w in open(path, "r").readlines()]
        self.multi_label = eval(self.labels.pop(0).split(':')[-1])
        assert isinstance(self.multi_label, bool), 'Bad formatting'
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.idx2label = {i: l for i, l in enumerate(self.labels)}
        return self


def process_hf_dataset(datasets, hf_local: bool, hf_filter, hf_field: str, split: str) -> List:
    # Normalize both datasets and hf_filter to lists
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(hf_filter, str):
        hf_filter = [hf_filter]
    elif hf_filter is None:
        hf_filter = []

    def process_single_dataset(dataset_name):
        load_func = load_from_disk if hf_local else load_dataset
        dataset = load_func(dataset_name)
        dataset = dataset[split]

        # Remove images column if different than hf_field (helps speed up filtering)
        first_example = dataset[0]
        field_pillow_image = [
            column_name
            for column_name in dataset.column_names
            if column_name != hf_field and (
                    isinstance(first_example[column_name], Image.Image) or
                    (isinstance(first_example[column_name], list) and first_example[column_name] and isinstance(
                        first_example[column_name][0], Image.Image))
            )
        ]

        dataset = dataset.remove_columns(field_pillow_image)

        # Filtering
        for fil in hf_filter:
            dataset = dataset.filter(eval(fil))

        # Selecting field
        dataset = dataset.select_columns([hf_field])
        return [d[hf_field] for d in tqdm.tqdm(dataset, desc=f"Loading {hf_field}")]

    results = []
    for dataset_name in datasets:
        results.extend(process_single_dataset(dataset_name))
    return results
