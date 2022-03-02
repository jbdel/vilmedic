import torch
from torch.nn import Dropout, Linear
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModel, AutoTokenizer


def data_cuda(*tensors, device='gpu', non_blocking=False):
    if device == 'gpu':
        cuda_tensors = []
        for tensor in tensors:
            cuda_tensors.append(tensor.cuda(non_blocking=non_blocking))
    else:
        cuda_tensors = tensors

    if len(cuda_tensors) > 1:
        return cuda_tensors
    else:
        return cuda_tensors[0]


class BERTNLI(torch.nn.Module):
    LABEL_CONTRADICTION = 2
    LABEL_ENTAILMENT = 0
    LABEL_NEUTRAL = 1

    def __init__(self, name_or_path, bert_type='bert', cls='linear', length=128, force_lowercase=False, device='cpu',
                 verbose=False):
        super(BERTNLI, self).__init__()
        self.bert_type = bert_type
        self.cls = cls
        self.force_lowercase = force_lowercase
        self.bert = AutoModel.from_pretrained(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.dropout = Dropout(0.1)
        self.linear = Linear(self.bert.config.hidden_size, 3)
        self.length = length
        self.device = device
        self.verbose = verbose

    @classmethod
    def train_step(cls, logits, gold_labels, optimizer, model=None, grad_clip=None):
        optimizer.zero_grad()
        target = logits.new_zeros((logits.shape[0],), dtype=torch.long)
        for i, gold_label in enumerate(gold_labels):
            if gold_label == 'entailment':
                target[i] = cls.LABEL_ENTAILMENT
            elif gold_label == 'neutral':
                target[i] = cls.LABEL_NEUTRAL
            elif gold_label == 'contradiction':
                target[i] = cls.LABEL_CONTRADICTION
            else:
                raise ValueError('Unknown label {0}'.format(gold_label))
        loss = cross_entropy(logits, target)
        loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach().cpu())

    def cuda(self, device=None):
        super(BERTNLI, self).cuda(device)
        if device != torch.device('cpu'):
            self.device = 'gpu'

    def forward(self, sent1s, sent2s):
        buffer, boundaries, max_len = [], [], 0
        for sent1, sent2 in zip(sent1s, sent2s):
            if self.force_lowercase:
                sent1 = sent1.lower()
                sent2 = sent2.lower()
            toks1 = self.tokenizer.tokenize(sent1)
            toks2 = self.tokenizer.tokenize(sent2)
            tokens = ['[CLS]'] + toks1 + ['[SEP]'] + toks2 + ['[SEP]']
            buffer.append(tokens)
            boundaries.append(len(toks1) + 2)
            if len(tokens) > max_len:
                max_len = len(tokens)
        if max_len > self.length:
            max_len = self.length
        token_ids, attn_mask = [], []
        seg_ids = [] if self.bert_type != 'distilbert' else None
        for idx, tokens in enumerate(buffer):
            if len(tokens) < max_len:
                for _ in range(max_len - len(tokens)):
                    tokens.append('[PAD]')
            elif len(tokens) > max_len:
                if self.verbose:
                    print('Truncating pair from {0}->{1}'.format(len(tokens), max_len))
                tokens = tokens[:max_len]
            attn_mask.append(torch.tensor([1 if token != '[PAD]' else 0 for token in tokens]))
            token_ids.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)))
            if seg_ids is not None:
                seg_ids.append(torch.tensor([0 if i < boundaries[idx] else 1 for i in range(len(tokens))]))
        token_ids = torch.stack(token_ids, dim=0)
        attn_mask = torch.stack(attn_mask, dim=0)
        if seg_ids is not None:
            seg_ids = torch.stack(seg_ids, dim=0)
            token_ids, attn_mask, seg_ids = data_cuda(token_ids, attn_mask, seg_ids, device=self.device)
        else:
            token_ids, attn_mask = data_cuda(token_ids, attn_mask, device=self.device)
        out = self.bert(token_ids.cuda(), attention_mask=attn_mask.cuda(), token_type_ids=seg_ids.cuda())
        reps = out["last_hidden_state"]
        cls = out["pooler_output"]
        if self.cls == 'token':
            reps = reps[:, 0]
            reps = self.dropout(reps)
            return self.linear(reps)
        else:
            cls = self.dropout(cls)
            return self.linear(cls)
