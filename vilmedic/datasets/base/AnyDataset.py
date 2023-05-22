import os
import json
import torch
import ast
from torch.utils.data import Dataset


def make_lines(root, split, file, processing):
    return [processing(line.strip()) for line in open(os.path.join(root, split + '.' + file))]


class AnyDataset(Dataset):
    def __init__(self,
                 root=None,
                 file=None,
                 split=None,
                 processing=None,
                 name=None,
                 **kwargs):
        assert split is not None, "Argument split cannot be None"

        self.root = root
        self.file = file
        self.split = split
        self.name = name or "any"
        self.processing = eval(processing or 'lambda x: x')
        self.lines = make_lines(root, split, file, self.processing)

    def __getitem__(self, index):
        return {self.name: self.lines[index]}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {self.name: [s[self.name] for s in batch]}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.lines)

    def inference(self, sentences):
        raise NotImplementedError()

    def __repr__(self):
        return "AnyDataset\n" + \
            json.dumps({"root": self.root,
                        "file": self.file,
                        "processing": self.processing,
                        "name": self.name
                        }, indent=4,
                       sort_keys=False, default=str)
