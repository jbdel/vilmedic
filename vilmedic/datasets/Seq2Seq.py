from torch.utils.data import Dataset
from .base.TextDataset import TextDataset


class Seq2Seq(Dataset):
    def __init__(self, src, tgt, split, ckpt_dir, **kwargs):
        self.split = split
        self.src = TextDataset(**src, split=split, ckpt_dir=ckpt_dir, source="src")
        self.tgt = TextDataset(**tgt, split=split, ckpt_dir=ckpt_dir, source="tgt")

        # For decoding
        self.tgt_tokenizer = self.tgt.tokenizer
        self.tgt_len = self.tgt.max_len

        # For tokenizing
        self.src_tokenizer_args = self.src.tokenizer_args
        self.tgt_tokenizer_args = self.tgt.tokenizer_args

        assert len(self.src) == len(self.tgt)

    def __getitem__(self, index):
        return {
            'src': ' '.join(self.src.__getitem__(index)[:self.src.max_len]),
            'tgt': ' '.join(self.tgt.__getitem__(index)[:self.tgt.max_len]) if self.split == 'train'
            else ' '.join(self.tgt.__getitem__(index))  # Never slice GT at eval/test time
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            src = self.src.tokenizer([s['src'] for s in batch], **self.src.tokenizer_args)
            tgt = self.tgt.tokenizer([s['tgt'] for s in batch], **self.tgt.tokenizer_args)
            collated = {'input_ids': src.input_ids,
                        'attention_mask': src.attention_mask,
                        'decoder_input_ids': tgt.input_ids,
                        'decoder_attention_mask': tgt.attention_mask}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.src)

    def __repr__(self):
        return "Seq2Seq\n" + str(self.src) + '\n' + str(self.tgt)
