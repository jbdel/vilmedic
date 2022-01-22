import torch
import tqdm
import os
import logging

from vilmedic.constants import EXTRA_CACHE_DIR

import torch.nn as nn
import pandas as pd

from collections import OrderedDict

from transformers import BertTokenizer
from transformers import BertModel, AutoModel, AutoConfig
from sklearn.metrics import classification_report
import numpy as np

logging.getLogger("urllib3").setLevel(logging.ERROR)


def generate_attention_masks(batch, source_lengths):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.cuda()


class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None, inference=False):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif inference:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape
                                            (batch_size, 4) and the last has shape (batch_size, 2)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


def tokenize(impressions, tokenizer):
    imp = impressions.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    impressions = imp.str.strip()
    new_impressions = []
    for i in (range(impressions.shape[0])):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512:  # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions


class CheXbert(nn.Module):
    def __init__(self, refs_filename=None, hyps_filename=None):
        super(CheXbert, self).__init__()
        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = bert_labeler(inference=True)
        state_dict = torch.load(os.path.join(EXTRA_CACHE_DIR, "chexbert.pth"))['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.cuda().eval()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.target_names = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices", "No Finding"]

        self.target_names_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        self.target_names_5_index = np.where(np.isin(self.target_names, self.target_names_5))[0]

    def get_label(self, report, mode="rrg"):
        impressions = pd.Series([report])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len)
        out = self.model(batch.cuda(), attn_mask)
        out = [out[j].argmax(dim=1).item() for j in range(len(out))]
        v = []
        if mode == "rrg":
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
            v = [1 if (isinstance(l, int) and l > 0) else 0 for l in v]

        elif mode == "classification":
            # https://github.com/stanfordmlgroup/CheXbert/blob/master/src/label.py#L124
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(-1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
        else:
            raise NotImplementedError(mode)

        return v

    def forward(self, hyps, refs):
        if self.refs_filename is None:
            refs_chexbert = [self.get_label(l.strip()) for l in refs]
        else:
            if os.path.exists(self.refs_filename):
                refs_chexbert = [eval(l.strip()) for l in open(self.refs_filename).readlines()]
            else:
                refs_chexbert = [self.get_label(l.strip()) for l in refs]
                open(self.refs_filename, 'w').write('\n'.join(map(str, refs_chexbert)))

        hyps_chexbert = [self.get_label(l.strip()) for l in hyps]
        if self.hyps_filename is not None:
            open(self.hyps_filename, 'w').write('\n'.join(map(str, hyps_chexbert)))

        refs_chexbert_5 = [np.array(r)[self.target_names_5_index] for r in refs_chexbert]
        hyps_chexbert_5 = [np.array(h)[self.target_names_5_index] for h in hyps_chexbert]

        cr = classification_report(refs_chexbert, hyps_chexbert, target_names=self.target_names, output_dict=True)
        cr_5 = classification_report(refs_chexbert_5, hyps_chexbert_5, target_names=self.target_names_5,
                                     output_dict=True)

        return cr, cr_5

    def train(self, mode: bool = True):
        mode = False  # force False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
