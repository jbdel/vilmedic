import os
import itertools


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
