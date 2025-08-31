import os
import tqdm
import json
import sys

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .utils import Vocab, load_file, process_hf_dataset
from .papers.report_preprocessing import *

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # https://github.com/huggingface/transformers/issues/5486


def split_sentences(sentences, processing):
    return [processing(s.strip()).split() for s in sentences]


def make_sentences(root, split, file, processing):
    sentences = load_file(os.path.join(root, split + '.' + file))
    return split_sentences(sentences, processing)


class TextDataset(Dataset):
    def __init__(self,
                 root=None,
                 file=None,
                 split=None,
                 ckpt_dir=None,
                 processing=None,
                 tokenizer=None,
                 tokenizer_max_len=None,
                 vocab_file=None,
                 source='src',
                 show_length=False,
                 hf_dataset=None,
                 hf_field=None,
                 hf_local=None,
                 hf_filter=None,
                 **kwargs):

        assert source in ["src", "tgt"]
        assert split is not None, "Argument split cannot be None"
        assert not (file is not None and vocab_file is not None), "You cannot mention both a data file and a vocab file"
        assert not (
                vocab_file is not None and tokenizer is not None), "You cannot mention both a pretrained tokenizer and a vocab file"
        assert not (source == "tgt" and tokenizer_max_len is None), "You must specify tokenizer_max_len for source tgt"
        assert (vocab_file is not None) or ((hf_dataset is None) != (
                file is None)), "Either 'hf_dataset' or 'file' must be specified, but not both or neither, unless 'vocab_file' is not None."

        assert hf_dataset is None or (hf_field is not None), "If 'hf_dataset' is not None, " \
                                                             "then 'hf_field' must also " \
                                                             "be not None."
        self.root = root
        self.file = file
        self.split = split
        self.source = source
        self.ckpt_dir = ckpt_dir
        self.processing = eval(processing or 'lambda x: x')
        self.tokenizer_max_len = tokenizer_max_len
        self.vocab_file = vocab_file
        self.sentences = None

        if file is not None:
            self.sentences = make_sentences(root, split, file, self.processing)

        if hf_dataset is not None:
            dataset = process_hf_dataset(hf_dataset, hf_local, hf_filter, hf_field, split)
            # Extract text from dataset and process
            sentences = [d[hf_field] for d in tqdm.tqdm(dataset, desc=f"Loading {hf_field}")]
            self.sentences = split_sentences(
                sentences=sentences,
                processing=self.processing
            )

        # Create tokenizer from pretrained or vocabulary file
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            if vocab_file is None:
                vocab_file = os.path.join(ckpt_dir, 'vocab.{}'.format(source))
                if split == 'train':
                    vocab = Vocab(self.sentences)
                    if not os.path.exists(vocab_file):
                        vocab.dump(vocab_file)
            self.tokenizer = BertTokenizer(vocab_file=vocab_file,
                                           do_basic_tokenize=False)

        # Create tokenizer forwards args
        self.tokenizer_args = {'return_tensors': 'pt', 'padding': True, 'add_special_tokens': True}
        if self.source == 'src':
            self.tokenizer_args.update({'add_special_tokens': False})
        if self.tokenizer_max_len is not None:
            self.tokenizer_args.update({'padding': 'max_length',
                                        'truncation': True,
                                        'max_length': self.tokenizer_max_len})

        if show_length:
            print(self)
            self.show_length()
            sys.exit()

    def __getitem__(self, index):
        return {'{}_seq'.format(self.source): ' '.join(self.sentences[index])}

    def get_collate_fn(self):
        def collate_fn(batch):
            seq = self.tokenizer([s['{}_seq'.format(self.source)] for s in batch], **self.tokenizer_args)
            collated = {
                'input_ids': seq.input_ids,
                'attention_mask': seq.attention_mask
            }
            return collated

        return collate_fn

    def __len__(self):
        return len(self.sentences or [])

    def __repr__(self):
        return "TextDataset\n" + \
            json.dumps({"source": self.source,
                        "root": self.root,
                        "file": self.file,
                        "len": len(self),
                        "processing": self.processing,
                        "Tokenizer": {
                            "name_or_path": self.tokenizer.name_or_path,
                            "vocab_size": self.tokenizer.vocab_size,
                            "tokenizer_args": self.tokenizer_args,
                            "special_tokens": self.tokenizer.special_tokens_map_extended,
                        }}, indent=4,
                       sort_keys=False, default=str)

    def show_length(self):
        print("Plotting sequences length...")
        sentence_len, tokenizer_len = [], []
        self.tokenizer_args.update({'max_length': 512})
        for index in tqdm.tqdm(range(len(self))):
            sentence = self.__getitem__(index)['{}_seq'.format(self.source)]
            x = self.tokenizer(sentence, **self.tokenizer_args).input_ids[0]
            if self.tokenizer.sep_token_id in x:
                length = ((x == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]).item()
            elif self.tokenizer.pad_token_id in x:
                length = ((x == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]).item()
            else:
                length = len(x)
            tokenizer_len.append(length)
            sentence_len.append(len(sentence.split()))

        _, ax = plt.subplots(1, 2)
        sns.histplot(tokenizer_len, ax=ax[0], binwidth=2)
        sns.histplot(sentence_len, ax=ax[1], binwidth=2)
        ax[0].set_title("tokenizer_len")
        ax[1].set_title("sentence_len")
        plt.show()
        plt.savefig(os.path.join(self.ckpt_dir, 'length.{}.png'.format(self.source)))
