import os
import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer
from .utils_hf import load_hf_dataset
try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:
    wordpunct_tokenize = None


def split_sentences(sentences, processing):
    return [processing(s.strip()).split() for s in sentences]


def ifcc_clean_report(report):
    report = report.lower()
    if wordpunct_tokenize is None:
        return ' '.join(report.split())
    return ' '.join(wordpunct_tokenize(report))


class TextDataset(Dataset):
    def __init__(self,
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
        assert hf_dataset is not None and hf_field is not None, "HF-only TextDataset requires hf_dataset and hf_field"

        self.split = split
        self.source = source
        self.ckpt_dir = ckpt_dir
        self.processing = eval(processing or 'lambda x: x')
        self.tokenizer_max_len = tokenizer_max_len
        self.vocab_file = vocab_file

        dataset = load_hf_dataset(hf_dataset, hf_local=hf_local, hf_filter=hf_filter, hf_field=hf_field, split=split)
        sentences = [d[hf_field] for d in tqdm.tqdm(dataset, desc=f"Loading {hf_field}")]
        self.sentences = split_sentences(sentences, self.processing)

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            # HF-only fallback: build/load vocab file and use BertTokenizer
            if vocab_file is None:
                assert ckpt_dir is not None, "ckpt_dir must be provided to generate vocab file"
                vocab_file = os.path.join(ckpt_dir, f'vocab.{source}')
                if split == 'train':
                    # Build vocab from tokenized sentences
                    words = set()
                    for tokens in self.sentences:
                        for t in tokens:
                            words.add(t)
                    special = ['[CLS]', '[PAD]', '[SEP]', '[UNK]', '[MASK]']
                    vocab_list = special + sorted(words)
                    if not os.path.exists(vocab_file):
                        tmp_path = vocab_file + '.tmp'
                        with open(tmp_path, 'w') as f:
                            f.write("\n".join(vocab_list))
                        os.replace(tmp_path, vocab_file)
            # Instantiate tokenizer from vocab
            self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_basic_tokenize=False)

        self.tokenizer_args = {'return_tensors': 'pt', 'padding': True, 'add_special_tokens': True}
        if self.source == 'src':
            self.tokenizer_args.update({'add_special_tokens': False})
        if self.tokenizer_max_len is not None:
            self.tokenizer_args.update({'padding': 'max_length', 'truncation': True, 'max_length': self.tokenizer_max_len})

    def __getitem__(self, index):
        return {f'{self.source}_seq': ' '.join(self.sentences[index])}

    def get_collate_fn(self):
        def collate_fn(batch):
            seq = self.tokenizer([s[f'{self.source}_seq'] for s in batch], **self.tokenizer_args)
            return {'input_ids': seq.input_ids, 'attention_mask': seq.attention_mask}
        return collate_fn

    def __len__(self):
        return len(self.sentences or [])

    def __repr__(self):
        return (
            "TextDataset(HF-only)\n" +
            f"source={self.source}, len={len(self)}, tokenizer={self.tokenizer.name_or_path}, max_len={self.tokenizer_max_len}"
        )


