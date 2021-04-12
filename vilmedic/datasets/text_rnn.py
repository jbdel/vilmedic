import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import Vocab


def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return [s for s in content.split('\n')]


def make_samples(root, split, src, tgt):
    src = load_file(os.path.join(root, split + '.' + src))
    tgt = load_file(os.path.join(root, split + '.' + tgt))

    src = [s.strip().split() for s in src]
    tgt = [t.strip().split() for t in tgt]

    return list(zip(src, tgt))


class TextDatasetRNN(Dataset):
    def __init__(self, root, split, ckpt_dir, src, tgt, max_len=80, **kwargs):
        self.root = root
        self.split = split
        self.samples = make_samples(root, split, src, tgt)

        src_vocab_file = os.path.join(ckpt_dir, 'vocab.src')
        tgt_vocab_file = os.path.join(ckpt_dir, 'vocab.tgt')

        if split == 'train':
            # Create vocab
            self.src_vocab = Vocab(map(lambda x: x[0], self.samples))
            self.tgt_vocab = Vocab(map(lambda x: x[1], self.samples))
            if not os.path.exists(src_vocab_file):
                self.src_vocab.dump(src_vocab_file)
                self.tgt_vocab.dump(tgt_vocab_file)

            self.src_len, self.tgt_len = max_len, max_len

            print('src_vocab', self.src_vocab)
            print('tgt_vocab', self.tgt_vocab)
            print('train_max_len', max_len)
        else:
            self.src_vocab = Vocab().load(src_vocab_file)
            self.tgt_vocab = Vocab().load(tgt_vocab_file)
            self.src_len, self.tgt_len = max_len, 200  # never trunc GT target sentence at test time

        self.processed_samples = []
        self.lengths = []

        self.tgt_trunc = 0
        self.src_trunc = 0

        for idx in range(len(self.samples)):
            src, tgt = self.samples[idx]

            self.src_trunc += int(len(src) > self.src_len)
            self.tgt_trunc += int(len(tgt) > self.tgt_len)

            src = src[:self.src_len] + ['[SEP]']
            tgt = ['[CLS]'] + tgt[:self.tgt_len] + ['[SEP]']
            self.processed_samples.append((
                torch.tensor(self.src_vocab.words2idxs(src)).long(),
                torch.tensor(self.tgt_vocab.words2idxs(tgt)).long())
            )
            self.lengths.append(len(src))

        print("{}: {} source samples at len > {}".format(split, self.src_trunc, self.src_len))
        print("{}: {} target samples at len > {}".format(split, self.tgt_trunc, self.tgt_len))

    def __getitem__(self, index):
        src, tgt = self.processed_samples[index]
        return {
            'src': src,
            'tgt': tgt,
        }

    def __len__(self):
        return len(self.processed_samples)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {
                'src': pad_sequence([s['src'] for s in batch], batch_first=False),
                'tgt': pad_sequence([s['tgt'] for s in batch], batch_first=False)}
            return collated

        return collate_fn
