import torch.nn as nn
from rouge_score import rouge_scorer
from six.moves import zip_longest
import numpy as np


class Rouge(nn.Module):
    def __init__(self, rouges, **kwargs):
        super().__init__()
        rouges = [r.replace('rougel', 'rougeL') for r in rouges]
        self.scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        self.rouges = rouges

    def forward(self, refs, hyps):
        scores = []
        for target_rec, prediction_rec in zip_longest(refs, hyps):
            if target_rec is None or prediction_rec is None:
                raise ValueError("Must have equal number of lines across target and "
                                 "prediction.")
            scores.append(self.scorer.score(target_rec, prediction_rec))
        f1_rouge = [s[self.rouges[0]].fmeasure for s in scores]
        return np.mean(f1_rouge), f1_rouge


class Rouge1(Rouge):
    def __init__(self, **kwargs):
        super(Rouge1, self).__init__(rouges=['rouge1'])


class Rouge2(Rouge):
    def __init__(self, **kwargs):
        super(Rouge2, self).__init__(rouges=['rouge2'])


class RougeL(Rouge):
    def __init__(self, **kwargs):
        super(RougeL, self).__init__(rouges=['rougeL'])
