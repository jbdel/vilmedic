#!/usr/bin/env python
# 
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import torch.nn as nn
from .bleu_scorer import BleuScorer


class Bleu(nn.Module):
    def __init__(self, n=4, **kwargs):
        # default compute Blue score up to 4
        super().__init__()
        self._n = n

    def forward(self, gts, res):
        return self.compute_score(gts, res)

    def compute_score(self, gts, res):
        res = {i: [v] for i, v in enumerate(res)}
        gts = {i: [v] for i, v in enumerate(gts)}
        bleu_scorer = BleuScorer(n=self._n)

        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score[self._n-1], scores[self._n-1]

    def method(self):
        return "Bleu"
