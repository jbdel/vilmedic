import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer
from ..text_rnn import make_samples
from ..utils import Vocab


class TextDatasetHug(Dataset):
    def __init__(self, root, split, ckpt_dir, src, tgt, max_len=80, tokenizer=None, **kwargs):
        self.root = root
        self.split = split
        self.src = src
        self.tgt = tgt
        self.ckpt_dir = ckpt_dir

        self.samples = make_samples(root, split, src, tgt)

        # If tokenizer exists, skip vocabulary creation
        if tokenizer is not None:
            self.src_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tgt_tokenizer = self.src_tokenizer
        else:
            src_vocab_file = os.path.join(ckpt_dir, 'vocab.src')
            tgt_vocab_file = os.path.join(ckpt_dir, 'vocab.tgt')
            if split == 'train':
                # Create vocab
                src_vocab = Vocab(map(lambda x: x[0], self.samples))
                tgt_vocab = Vocab(map(lambda x: x[1], self.samples))
                if not os.path.exists(src_vocab_file):
                    src_vocab.dump(src_vocab_file)
                    tgt_vocab.dump(tgt_vocab_file)
                print('src_vocab', src_vocab)
                print('tgt_vocab', tgt_vocab)

            self.src_tokenizer = BertTokenizer(vocab_file=src_vocab_file, do_basic_tokenize=False)
            self.tgt_tokenizer = BertTokenizer(vocab_file=tgt_vocab_file, do_basic_tokenize=False)

        self.src_len, self.tgt_len = (max_len, max_len)
        if split == 'train':
            print('train_max_len', max_len)

    def __getitem__(self, index):
        src, tgt = self.samples[index]
        return {
            'src': ' '.join(src[:self.src_len]),
            'tgt': ' '.join(tgt[:self.tgt_len]) if self.split == 'train' else ' '.join(tgt)
            # Never slice GT at eval/test time
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            src = self.src_tokenizer([s['src'] for s in batch], padding=True, return_tensors="pt",
                                     add_special_tokens=False)
            tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, return_tensors="pt")
            collated = {'input_ids': src.input_ids,
                        'attention_mask': src.attention_mask,
                        'decoder_input_ids': tgt.input_ids,
                        'decoder_attention_mask': tgt.attention_mask}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.samples)
