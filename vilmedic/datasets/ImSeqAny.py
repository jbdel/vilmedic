from torch.utils.data import Dataset
from .base.AnyDataset import AnyDataset
from .ImSeq import ImSeq


class ImSeqAny(Dataset):
    def __init__(self, seq, any, image, split, ckpt_dir, **kwargs):
        self.split = split
        self.imgseq = ImSeq(seq, image, split=split, ckpt_dir=ckpt_dir)
        self.any = AnyDataset(**any, split=split, ckpt_dir=ckpt_dir)

        assert len(self.imgseq) == len(self.any), str(len(self.imgseq)) + 'vs ' + str(len(self.any))

        # For models
        self.seq = self.imgseq.seq
        self.image = self.imgseq.image

        # For decoding, if needed
        self.tokenizer = self.imgseq.seq.tokenizer
        self.tokenizer_max_len = self.imgseq.seq.tokenizer_max_len

        # For tokenizing
        self.tokenizer_args = self.imgseq.seq.tokenizer_args

    def __getitem__(self, index):
        return {**self.imgseq.__getitem__(index), **self.any.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {**self.imgseq.get_collate_fn()(batch),
                        **self.any.get_collate_fn()(batch)}
            return collated

        return collate_fn

    def __len__(self):
        return len(self.any)

    def __repr__(self):
        return "ImSeqAny\n" + str(self.imgseq) + '\n' + str(self.any)
