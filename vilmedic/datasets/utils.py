class Vocab():
    extra = None

    def __init__(self, sentences=None,
                 pad_token="[PAD]",
                 eos_token="[SEP]",
                 bos_token="[CLS]",
                 unk_token="[UNK]",
                 mask_token="[MASK]"):

        if sentences is not None:
            self.vocab = []
            for sentence in sentences:
                self.vocab += sentence
            self.vocab = sorted(set(self.vocab))
            self.extra = [pad_token, bos_token, eos_token, unk_token, mask_token]

            self.words = self.extra + self.vocab
            self.inv_words = {w: i for i, w in enumerate(self.words)}
            assert self.words[:len(self.extra)] == self.extra

    def get_eos(self):
        return self.extra[2]

    def get_bos(self):
        return self.extra[1]

    def get_unk(self):
        return self.extra[3]

    def strip_beos_w(self, words):
        if words[0] == self.get_bos():
            del words[0]

        result = []
        for w in words:
            if w == self.get_eos():
                break
            result.append(w)
        return result

    def word2idx(self, word):
        if word not in self.inv_words:
            word = self.get_unk()
        return self.inv_words[word]

    def idx2word(self, idx):
        if idx < len(self):
            return self.words[idx]
        else:
            return self.get_unk()

    def idxs2words(self, idxs):
        return [self.idx2word(idx) for idx in idxs]

    def words2idxs(self, words):
        return [self.word2idx(word) for word in words]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return "Vocab(#words={}, #vocab={}, #extra={})".format(len(self), len(self.vocab), len(self.extra))

    def __iter__(self):
        for w in self.words:
            yield w

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.words))

    def load(self, path):
        self.words = [w.strip() for w in open(path, "r").readlines()]
        self.vocab = self.words[5:]
        self.extra = self.words[:5:]
        self.inv_words = {w: i for i, w in enumerate(self.words)}
        return self


class Labels(object):
    def __init__(self, answers=None):
        if answers is not None:
            self.labels = set(answers)
            self.inv_labels = {l: i for i, l in enumerate(self.labels)}

    def label2idx(self, label):
        return self.inv_labels[label]

    def idx2label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return "Label(#labels={})".format(len(self))

    def dump(self, path):
        open(path, "w").write("\n".join(str(w) for w in self.labels))

    def load(self, path):
        self.labels = [w.strip() for w in open(path, "r").readlines()]
        self.inv_labels = {l: i for i, l in enumerate(self.labels)}
        return self
