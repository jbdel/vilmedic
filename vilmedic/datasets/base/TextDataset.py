import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .utils import Vocab, load_file


def make_sentences(root, split, file):
    sentences = load_file(os.path.join(root, split + '.' + file))
    return [s.strip().split() for s in sentences]


class TextDataset(Dataset):
    def __init__(self, root, file, split, ckpt_dir, source='src', max_len=80, tokenizer=None, tokenizer_max_len=None,
                 **kwargs):

        self.root = root
        self.split = split
        self.source = source
        self.ckpt_dir = ckpt_dir
        self.max_len = max_len
        self.tokenizer_max_len = tokenizer_max_len

        self.sentences = make_sentences(root, split, file)

        # If tokenizer exists, skip vocabulary creation
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            # Get vocab
            vocab_file = os.path.join(ckpt_dir, 'vocab.{}'.format(source))
            if split == 'train':
                vocab = Vocab(self.sentences)
                if not os.path.exists(vocab_file): vocab.dump(vocab_file)
                print('{}_vocab'.format(source), vocab)

            self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_basic_tokenize=False)

        if split == 'train':
            print('Training {} with max_len {}'.format(self.source, self.max_len))

    def __getitem__(self, index):
        # ['w1', 'w2', 'w3']
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)
