from torch.utils.data import Dataset
from .base.TextDataset import TextDataset


class Seq2Seq(Dataset):
    def __init__(self, src, tgt, split, ckpt_dir, **kwargs):
        self.split = split
        self.src = TextDataset(**src, split=split, ckpt_dir=ckpt_dir, source="src")
        self.tgt = TextDataset(**tgt, split=split, ckpt_dir=ckpt_dir, source="tgt")

        # For decoding
        self.tgt_tokenizer = self.tgt.tokenizer
        self.tgt_tokenizer_max_len = self.tgt.tokenizer_max_len

        assert len(self.src) == len(self.tgt)

    def __getitem__(self, index):
        return {**self.src.__getitem__(index), **self.tgt.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            tgt = self.tgt.get_collate_fn()(batch)
            tgt['decoder_input_ids'] = tgt.pop('input_ids')
            tgt['decoder_attention_mask'] = tgt.pop('attention_mask')
            collated = {**self.src.get_collate_fn()(batch), **tgt}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.src)

    def __repr__(self):
        return "Seq2Seq\n" + str(self.src) + '\n' + str(self.tgt)

    def inference(self, src=None, tgt=None):
        if src is None and tgt is None:
            return dict()

        batch = {}
        if src is not None:
            batch.update(self.src.inference(src))

        if tgt is not None:
            tgt = self.tgt.inference(tgt)
            tgt['decoder_input_ids'] = tgt.pop('input_ids')
            tgt['decoder_attention_mask'] = tgt.pop('attention_mask')
            batch.update(tgt)

        assert len(set([len(v) for k, v in batch.items()])) == 1, 'elements in batch do not have the same size'
        return batch
