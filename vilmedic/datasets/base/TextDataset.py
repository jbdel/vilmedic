import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .utils import Vocab, load_file
import json


def make_sentences(root, split, file):
    sentences = load_file(os.path.join(root, split + '.' + file))
    return [s.strip().split() for s in sentences]


class TextDataset(Dataset):
    def __init__(self, root, file, split, ckpt_dir, source='src', max_len=250, tokenizer=None, tokenizer_max_len=None,
                 **kwargs):

        assert source in ["src", "tgt"]

        self.root = root
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

    def __getitem__(self, index):
        return self.sentences[index]  # ['w1', 'w2', 'w3']

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return "TextDataset\n" + \
               json.dumps({"source": self.source,
                           "max_len": self.max_len,
                           "Tokenizer": {
                               "name_or_path": self.tokenizer.name_or_path,
                               "vocab_size": self.tokenizer.vocab_size,
                               "tokenizer_args": self.tokenizer_args,
                               "special_tokens": self.tokenizer.special_tokens_map_extended}}, indent=4,
                          sort_keys=False, default=str)
