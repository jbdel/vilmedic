import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .utils import Vocab, load_file
import json
import sys


def make_sentences(root, split, file):
    sentences = load_file(os.path.join(root, split + '.' + file))
    return [s.strip().split() for s in sentences]


class TextDataset(Dataset):
    def __init__(self, root, file, split, ckpt_dir, source='src', max_len=250, tokenizer=None, tokenizer_max_len=None,
                 show_length=False,
                 **kwargs):

        assert source in ["src", "tgt"]

        self.root = root
        self.file = file
        self.split = split
        self.source = source
        self.ckpt_dir = ckpt_dir
        self.max_len = max_len
        self.tokenizer_max_len = tokenizer_max_len

        self.sentences = make_sentences(root, split, file)

        # Create tokenizer from pretrained or vocabulary file
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            vocab_file = os.path.join(ckpt_dir, 'vocab.{}'.format(source))
            if split == 'train':
                vocab = Vocab(self.sentences)
                if not os.path.exists(vocab_file): vocab.dump(vocab_file)
            self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_basic_tokenize=False)

        # Create tokenizer forwards args
        self.tokenizer_args = {'return_tensors': 'pt', 'padding': True}
        if self.source == 'src':
            self.tokenizer_args.update({'add_special_tokens': False})
        if self.tokenizer_max_len is not None:
            self.tokenizer_args.update({'padding': 'max_length',
                                        'truncation': True,
                                        'max_length': self.tokenizer_max_len})

        if show_length:
            self.show_length()

    def __getitem__(self, index):
        return self.sentences[index]  # ['w1', 'w2', 'w3']

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return "TextDataset\n" + \
               json.dumps({"source": self.source,
                           "root": self.root,
                           "file": self.file,
                           "max_len": self.max_len,
                           "Tokenizer": {
                               "name_or_path": self.tokenizer.name_or_path,
                               "vocab_size": self.tokenizer.vocab_size,
                               "tokenizer_args": self.tokenizer_args,
                               "special_tokens": self.tokenizer.special_tokens_map_extended}}, indent=4,
                          sort_keys=False, default=str)

    def show_length(self):
        import tqdm
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="darkgrid")

        sentence_len = []
        tokenizer_len = []

        for index in tqdm.tqdm(range(len(self))):
            sentence = self.__getitem__(index)
            x = self.tokenizer(' '.join(sentence), **self.tokenizer_args).input_ids[0]
            if self.tokenizer.sep_token_id in x:
                length = ((x == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]).item()
            elif self.tokenizer.pad_token_id in x:
                length = ((x == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]).item()
            else:
                length = len(x)
            tokenizer_len.append(length)
            sentence_len.append(len(sentence))

        _, ax = plt.subplots(1, 2)
        sns.histplot(tokenizer_len, ax=ax[0], binwidth=2)
        sns.histplot(sentence_len, ax=ax[1], binwidth=2)
        ax[0].set_title("tokenizer_len")
        ax[1].set_title("sentence_len")
        plt.show()
