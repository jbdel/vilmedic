import collections
import os
from pickle import NONE
import tqdm
import json
import sys
from itertools import chain, combinations

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .utils import Vocab, load_file
from .papers.report_preprocessing import *

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="darkgrid")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # https://github.com/huggingface/transformers/issues/5486


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def split_sentences(sentences, processing):
    # return [processing(s.strip()).split() for s in sentences]
    return ['' if s == "REMOVE" else processing(s.strip()).split() for s in sentences]


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
                 separate_tokenizer_per_phrase=False,
                 separate_tokenizer_per_phrase_delimiter=",",
                 num_concepts=None,
                 vocab_file=None,
                 source='src',
                 show_length=False,
                 name=None,
                 n_examples=None,
                 **kwargs):

        assert source in ["src", "tgt"]
        assert split is not None, "Argument split cannot be None"
        assert not (file is not None and vocab_file is not None), "You cannot mention both a data file and a vocab file"
        assert not (
                vocab_file is not None and tokenizer is not None), "You cannot mention both a pretrained tokenizer and a vocab file"
        # assert not (source == "tgt" and tokenizer_max_len is None), "You must specify tokenizer_max_len for source tgt"

        self.root = root
        self.file = file
        self.split = split
        self.source = source
        self.name = name if name else self.source
        self.ckpt_dir = ckpt_dir
        self.processing = eval(processing or 'lambda x: x')
        self.tokenizer_max_len = tokenizer_max_len
        self.num_concepts = num_concepts
        self.vocab_file = vocab_file
        self.sentences = None

        if file is not None:
            self.sentences = make_sentences(root, split, file, self.processing)

        if n_examples is not None:
            self.sentences = self.sentences[:n_examples]

        # Create tokenizer from pretrained or vocabulary file
        self.separate_tokenizer_per_phrase = separate_tokenizer_per_phrase
        self.separate_tokenizer_per_phrase_delimiter = separate_tokenizer_per_phrase_delimiter

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
        return {'{}_seq'.format(self.name): ' '.join(self.sentences[index])}

    def get_collate_fn(self):
        def collate_fn(batch):
            seq = self.tokenizer([s['{}_seq'.format(self.name)] for s in batch], **self.tokenizer_args)
            collated = {
                'input_ids': seq.input_ids,
                'attention_mask': seq.attention_mask
            }
            return collated

        def separate_tokenizer_per_phrase_collate_fn(batch):
            collated = {
                'input_ids': [],
            }

            phrase_seqs = [s['{}_seq'.format(self.name)] for s in batch]
            for phrase_str in phrase_seqs:
                phrase_list = phrase_str.split(self.separate_tokenizer_per_phrase_delimiter)
                phrase_list = [phrase.strip() for phrase in phrase_list]

                if self.num_concepts > 0:
                    np.random.shuffle(phrase_list)
                    phrase_list = phrase_list[:self.num_concepts]
                elif self.num_concepts == 0:
                    phrase_list = phrase_list
                else:
                    if phrase_list != ['']:
                        CHEXPERT_CONCEPTS = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion']
                        phrase_dict = collections.defaultdict(list)
                        for concept in CHEXPERT_CONCEPTS:
                            temp_phrase_list_for_concept = [phrase for phrase in phrase_list if concept in phrase]
                            if len(temp_phrase_list_for_concept) > 1:
                                # constrained decoding cannot have a single term in a disjunctive set be a complete subset of any other phrases
                                # therefore, just keep the single, smaller complete subset phrase!
                                remove_sets = []
                                keep_sets = []
                                for this_phrase in temp_phrase_list_for_concept:
                                    if any([other_phrase in this_phrase for other_phrase in temp_phrase_list_for_concept if this_phrase != other_phrase]):
                                        # remove the superset
                                        remove_sets.append(this_phrase)
                                    else:
                                        keep_sets.append(this_phrase)

                                phrase_dict[concept] = [phrase for phrase in keep_sets if phrase not in remove_sets]
                            else:
                                phrase_dict[concept] = temp_phrase_list_for_concept

                        phrase_list = [v for k,v in phrase_dict.items() if len(v) > 0]
                
                # input_ids_for_example = []
                # for phrase in phrase_list:
                #     # TODO: do we need add_prefix_space=True?
                #     # Yes! https://github.com/GXimingLu/a_star_neurologic/blob/main/translation/decode.py#L79
                #     phrase_input_ids = self.tokenizer(phrase, add_prefix_space=False, **self.tokenizer_args).input_ids
                #     # phrase_input_ids = phrase_input_ids[1:]
                #     input_ids_for_example.append(phrase_input_ids)
                input_ids_for_example = [self.tokenizer(phrase, add_prefix_space=True, add_special_tokens=False).input_ids for phrase in phrase_list]
                collated['input_ids'].append(input_ids_for_example)
            return collated

        return separate_tokenizer_per_phrase_collate_fn if self.separate_tokenizer_per_phrase else collate_fn

    def __len__(self):
        return len(self.sentences or [])

    def inference(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        batch = [{'{}_seq'.format(self.source): self.processing(s)} for s in sentences]
        return self.get_collate_fn()(batch)

    def __repr__(self):
        return "TextDataset\n" + \
               json.dumps({"source": self.source,
                           "root": self.root,
                           "file": self.file,
                           "processing": self.processing,
                           "separate_tokenizer_per_phrase": self.separate_tokenizer_per_phrase,
                           "separate_tokenizer_per_phrase_delimiter": self.separate_tokenizer_per_phrase_delimiter,
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
