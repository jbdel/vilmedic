import json
import random


class Vocab():
    extra = [
        "[PAD]",
        "[SEP]",
        "[CLS]",
        "[UNK]",
        "[MASK]"]

    def __init__(self, sentences=None):
        if sentences is not None:
            self.vocab = []
            for sentence in sentences:
                self.vocab += sentence
            self.vocab = sorted(set(self.vocab))
            # extra at first so that different vocabs share the same extra idxs
            self.words = Vocab.extra + self.vocab
            self.inv_words = {w: i for i, w in enumerate(self.words)}
            assert self.words[:len(self.extra)] == self.extra

    @staticmethod
    def extra2idx(e):
        return Vocab.extra.index(e)

    def strip_beos_w(self, words):
        if words[0] == '[CLS]':
            del words[0]

        result = []
        for w in words:
            if w == '[SEP]':
                break
            result.append(w)
        return result

    def word2idx(self, word):
        if word not in self.inv_words:
            word = '[UNK]'
        return self.inv_words[word]

    def idx2word(self, idx):
        if idx < len(self):
            return self.words[idx]
        else:
            return '[UNK]'

    def idxs2words(self, idxs):
        return [self.idx2word(idx) for idx in idxs]

    def words2idxs(self, words):
        return [self.word2idx(word) for word in words]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return "Vocab(#words={}, #vocab={}, #extra={})".format(len(self), len(self.vocab),
                                                               len(self.extra))

    def __iter__(self):
        for w in self.words:
            yield w

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.words))

    def load(self, path):
        self.words = [w.strip() for w in open(path, "r").readlines()]
        self.vocab = self.words[len(Vocab.extra):]
        self.inv_words = {w: i for i, w in enumerate(self.words)}
        return self

class Labels(object):
    def __init__(self, answers):
        self.labels = sorted(set(answers))
        self.inv_labels = {l: i for i, l in enumerate(self.labels)}

    def label2idx(self, label):
        return self.inv_labels[label]

    def idx2label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return "Label(#labels={})".format(len(self))
