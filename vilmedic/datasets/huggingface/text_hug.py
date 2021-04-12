import os
from torch.utils.data import Dataset
from transformers import BertTokenizer
from ..text_rnn import make_samples
from ..utils import Vocab


class TextDatasetHugStatic:
    src_len = None
    tgt_len = None


class TextDatasetHug(Dataset):
    def __init__(self, root, split, ckpt_dir, src, tgt, src_len=80, tgt_len=80, **kwargs):
        self.root = root
        self.split = split
        self.samples = make_samples(root, split, src, tgt)

        src_vocab_file = os.path.join(ckpt_dir, 'vocab.src')
        tgt_vocab_file = os.path.join(ckpt_dir, 'vocab.tgt')

        if split == 'train':
            src_vocab = Vocab(map(lambda x: x[0], self.samples))
            tgt_vocab = Vocab(map(lambda x: x[1], self.samples))

            src_vocab.dump(src_vocab_file)
            tgt_vocab.dump(tgt_vocab_file)

            TextDatasetHugStatic.src_len = src_len
            TextDatasetHugStatic.tgt_len = tgt_len

            print('src_vocab', src_vocab)
            print('tgt_vocab', tgt_vocab)
            print('src_len', src_len)
            print('tgt_len', tgt_len)

        self.src_len, self.tgt_len = TextDatasetHugStatic.src_len, TextDatasetHugStatic.tgt_len

        self.src_tokenizer = BertTokenizer(vocab_file=src_vocab_file, do_basic_tokenize=False)
        self.tgt_tokenizer = BertTokenizer(vocab_file=tgt_vocab_file, do_basic_tokenize=False)

    def __getitem__(self, index):
        src, tgt = self.samples[index]
        return {
            'src': ' '.join(src)[:self.src_len],
            'tgt': ' '.join(tgt)[:self.tgt_len],
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            src = self.src_tokenizer([s['src'] for s in batch], padding=True, return_tensors="pt", add_special_tokens=False)
            tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, return_tensors="pt")

            collated = {'input_ids': src.input_ids,
                        'attention_mask': src.attention_mask,
                        'decoder_input_ids': tgt.input_ids,
                        'decoder_attention_mask': tgt.attention_mask}

            # src = self.src_tokenizer([s['src'] for s in batch], padding=True, truncation=True, return_tensors="pt",
            #                          max_length=self.src_len)
            # tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, truncation=True, return_tensors="pt",
            #                          max_length=self.tgt_len)
            # v = batch[0]["tgt"]
            # print(len(v))
            # print(v)

            # print(src)
            # print(tgt)
            # sys.exit()
            # print("####")
            # print(tgt["input_ids"].shape)
            # print(tgt["input_ids"][0])
            # print(len(tgt["input_ids"][0]))
            # print(self.tgt_tokenizer.decode(tgt["input_ids"][0], skip_special_tokens=False, clean_up_tokenization_spaces=False))

            return collated

        return collate_fn

    def __len__(self):
        return len(self.samples)
