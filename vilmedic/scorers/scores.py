import os
import numpy as np
import torch.nn.functional as F
import torch

from .nlg import ROUGEScorer
from .nlg import METEORScorer
from .nlg import BLEUScorer

from sklearn.metrics import classification_report, roc_auc_score


def compute_scores(metrics, refs, hyps, split, seed, ckpt_dir, epoch):
    scores = {}
    if metrics is None or not metrics:
        return scores

    assert len(refs) == len(hyps), '{} vs {}'.format(len(refs), len(hyps))

    # Dump
    base = os.path.join(ckpt_dir, '{}_{}_{}'.format(split, seed, '{}'))
    refs_file = base.format('refs.txt')
    hyps_file = base.format('hyps.txt')
    metrics_file = base.format('metrics.txt')

    with open(refs_file, 'w') as f:
        f.write('\n'.join(map(str, refs)))

    with open(hyps_file, 'w') as f:
        f.write('\n'.join(map(str, hyps)))

    for metric in metrics:
        if metric == 'BLEU':
            scores["BLEU"] = round(BLEUScorer().compute(refs_file, hyps_file), 2)
        elif metric == 'ROUGE1':
            scores["ROUGE1"] = round(ROUGEScorer(rouges=['rouge1']).compute(refs, hyps) * 100, 2)
        elif metric == 'ROUGE2':
            scores["ROUGE2"] = round(ROUGEScorer(rouges=['rouge2']).compute(refs, hyps) * 100, 2)
        elif metric == 'ROUGEL':
            scores["ROUGEL"] = round(ROUGEScorer(rouges=['rougeL']).compute(refs, hyps) * 100, 2)
        elif metric == 'METEOR':
            scores["METEOR"] = round(METEORScorer().compute(refs_file, hyps_file) * 100, 2)
        elif metric == 'accuracy':
            scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
        elif metric == 'f1-score':
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == 'auroc':
            scores["auroc"] = roc_auc_score(refs, F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), multi_class="ovr")
        elif metric == 'chexbert':
            pass
        else:
            raise NotImplementedError(metric)

    with open(metrics_file, 'a+') as f:
        f.write(str({
            'split': split,
            'epoch': epoch,
            'scores': scores,
        }) + '\n')

    return scores
