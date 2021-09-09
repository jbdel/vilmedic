import os
import itertools


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


class Vocab:
    def __init__(self, sentences=None,
                 pad_token="[PAD]",
                 eos_token="[SEP]",
                 bos_token="[CLS]",
                 unk_token="[UNK]",
                 mask_token="[MASK]"):
        self.vocab = list(itertools.chain(*sentences))
        self.words = [pad_token, bos_token, eos_token, unk_token, mask_token] + sorted(set(self.vocab))

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.words))


class Labels:
    def __init__(self, answers=None):
        if answers is not None:
            self.labels = set(answers)
            self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.labels))

    def load(self, path):
        self.labels = [w.strip() for w in open(path, "r").readlines()]
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        return self
