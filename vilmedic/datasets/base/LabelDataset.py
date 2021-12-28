from torch.utils.data import Dataset
from torchvision.transforms import *
import os
from .utils import Labels, load_file
import json


def make_labels(root, split, file):
    return load_file(os.path.join(root, split + '.' + file))


class LabelDataset(Dataset):

    def __init__(self,
                 root=None,
                 split=None,
                 file=None,
                 ckpt_dir=None,
                 label_file=None,
                 **kwargs):

        assert split is not None, "Argument split cant be None"
        assert not (file is None and label_file is None), "Please specify a file or a label_file"

        self.root = root
        self.split = split
        self.label_file = label_file
        self.labels_map = None
        self.labels = None

        # Create a label file from data file
        if file is not None:
            self.labels = make_labels(root, split, file)
            self.label_file = os.path.join(ckpt_dir, 'labels.tok')
            if split == 'train' and not os.path.exists(self.label_file):
                Labels(self.labels).dump(self.label_file)

        # Load data file
        try:
            self.labels_map = Labels().load(self.label_file)
        except FileNotFoundError:
            raise FileNotFoundError("label file does not exists, verify path or start a training")

        # Compute labels according to data file and label file
        if file is not None:
            labels = []
            for label in self.labels:
                processed_label = self.get_processed_label(label)
                labels.append(processed_label)
            self.labels = labels

    def __len__(self):
        return len(self.labels or [])

    def __getitem__(self, index):
        return {'label': self.labels[index]}

    def inference(self, label):
        if not isinstance(label, list):
            label = [label]
        batch = [{'label': self.get_processed_label(l)} for l in label]
        return self.get_collate_fn()(batch)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'labels': torch.stack([s['label'] for s in batch])}
            return collated

        return collate_fn

    def get_processed_label(self, label):
        try:
            classes = [l for l in label.split(',')]
            if not self.labels_map.multi_label:  # single label
                return torch.tensor(self.labels_map.label2idx[classes[0]]).long()
            else:  # multi-label
                multi_hot = torch.zeros(len(self.labels_map.idx2label))
                multi_hot[[self.labels_map.label2idx[c] for c in classes]] = 1.
                return multi_hot
        except KeyError:
            # Label in a split is not present in train set, this can happen (OOD or so.)
            return torch.tensor(-100).long()

    def __repr__(self):
        return "LabelDataset\n" + \
               json.dumps({"num_labels": len(self.labels_map.labels)}, indent=4,
                          sort_keys=False, default=str)
