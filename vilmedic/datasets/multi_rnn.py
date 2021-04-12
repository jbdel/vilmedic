import os
import torch
from torch.nn.utils.rnn import pad_sequence
from .text_rnn import TextDatasetRNN
import numpy as np


class MultiDatasetRNN(TextDatasetRNN):
    def __init__(self, npy, npy_root, **kwargs):
        super().__init__(**kwargs)
        self.file = npy
        self.npy_root = npy_root
        self.features = np.load(os.path.join(self.npy_root, self.split + '.' + self.file))
        if 'npz' in self.file:
            self.features = self.features['arr_0']
        assert self.__len__() == super().__len__(), 'len(MultiDatasetRNN) != len(TextDatasetRNN)'

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        ret['feats'] = self.features[index]
        return ret

    def __len__(self):
        return len(self.features)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {
                'src': pad_sequence([s['src'] for s in batch], batch_first=False),
                'tgt': pad_sequence([s['tgt'] for s in batch], batch_first=False)}
            # Features
            v = torch.from_numpy(np.array(
                [s['feats'] for s in batch], dtype='float32'))

            # if 2D features (avg pool for eg)
            if len(v.size()) == 2:
                v = v.unsqueeze(1)

            # v is now (bs, n, feat)
            collated['feats'] = v
            return collated

        return collate_fn
