import os
import numpy as np
import json
import torch.nn.functional as F
import torch
# import nlgeval

from .NLG import ROUGEScorer
from .NLG import METEORScorer
from .NLG import BLEUScorer
from .NLG import Cider
import mauve
from .NLG import CiderD
# from nlgeval import NLGEval

from .CheXbert.chexbert import CheXbert

from sklearn.metrics import classification_report, roc_auc_score


def compute_scores(metrics, refs, hyps, split, seed, config, epoch):
    scores = dict()
    # If metric is None or empty list
    if metrics is None or not metrics:
        return scores

    assert refs is not None and hyps is not None, \
        "You specified metrics but your evaluation does not return hyps nor refs"

    assert len(refs) == len(hyps), 'refs and hyps must have same length : {} vs {}'.format(len(refs), len(hyps))

    # Dump
    ckpt_dir = config.ckpt_dir
    base = os.path.join(ckpt_dir, '{}_{}_{}'.format(split, seed, '{}'))
    refs_file = base.format('refs.txt')
    hyps_file = base.format('hyps.txt')
    metrics_file = base.format('metrics.txt')

    with open(refs_file, 'w') as f:
        f.write('\n'.join(map(str, refs)))
        f.close()

    with open(hyps_file, 'w') as f:
        f.write('\n'.join(map(str, hyps)))
        f.close()

    for metric in metrics:
        if metric == 'BLEU':
            scores["BLEU"] = round(BLEUScorer().compute(refs_file, hyps_file), 2)
        elif metric == 'ROUGE1':
            scores["ROUGE1"] = round(ROUGEScorer(rouges=['rouge1']).compute(refs, hyps)[0] * 100, 2)
        elif metric == 'ROUGE2':
            scores["ROUGE2"] = round(ROUGEScorer(rouges=['rouge2']).compute(refs, hyps)[0] * 100, 2)
        elif metric == 'ROUGEL':
            scores["ROUGEL"] = round(ROUGEScorer(rouges=['rougeL']).compute(refs, hyps)[0] * 100, 2)
        elif metric == 'METEOR':
            scores["METEOR"] = round(METEORScorer().compute(refs_file, hyps_file) * 100, 2)
        elif metric == 'CIDER':
            scores["CIDER"] = Cider().compute_score(refs, hyps)
        elif metric == 'MAUVE':
            scores["MAUVE"] = mauve.compute_mauve(p_text=refs, q_text=hyps, device_id=0, max_text_length=256,
                                                  featurize_model_name=config.mauve_featurize_model_name or "gpt2-large")
        elif metric == 'accuracy':
            scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
        elif metric == 'f1-score':
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == 'auroc':
            scores["auroc"] = roc_auc_score(refs, F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), multi_class="ovr")
        elif metric == 'chexbert':
            scores["chexbert"], scores["chexbert-5"] = CheXbert(refs_filename=base.format('refs.chexbert.txt'),
                                                                hyps_filename=base.format('hyps.chexbert.txt'))(hyps,
                                                                                                                refs)
            scores["chexbert-5_micro avg_f1-score"] = scores["chexbert-5"]["micro avg"]["f1-score"]
        else:
            raise NotImplementedError(metric)

    with open(metrics_file, 'a+') as f:
        f.write(json.dumps({
            'split': split,
            'epoch': epoch,
            'scores': scores
        }, indent=4, sort_keys=False))
    return scores
