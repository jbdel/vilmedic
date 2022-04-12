import os
import numpy as np
import json
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report, roc_auc_score
from omegaconf import OmegaConf
from . import *

REWARD_COMPLIANT = {
    "BLEU": BLEUScorer,
    "ROUGE1": ROUGEScorer,
    "ROUGE2": ROUGEScorer,
    "ROUGEL": BLEUScorer,
    "METEOR": METEORScorer,
    "CIDER": Cider,
    "MAUVE": MauveScorer,
    "radentitymatchexact": RadEntityMatchExact,
    "radentitynli": RadEntityNLI,
}


def compute_scores(metrics, refs, hyps, split, seed, config, epoch, logger):
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
        metric_args = dict()

        # if metric has arguments
        if OmegaConf.is_dict(metric):
            if len(metric) != 1:
                logger.warning("Metric badly formatted: {}. Skipping.".format(metric))
                continue
            metric_args = metric[list(metric.keys())[0]]
            metric = list(metric.keys())[0]

        # Iterating over metrics
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
            scores["CIDER"] = Cider(**metric_args).compute_score(refs, hyps)
        elif metric == 'CIDERD':
            scores["CIDERD"] = CiderD(**metric_args).compute_score(refs, hyps)
        elif metric == 'MAUVE':
            scores["MAUVE"] = round(
                MauveScorer(config.mauve_featurize_model_name or "distilgpt2").compute(refs, hyps) * 100, 2)
        elif metric == 'accuracy':
            scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
        elif metric == 'f1-score':
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == 'auroc':
            scores["auroc"] = roc_auc_score(refs, F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), multi_class="ovr")
        elif metric == 'chexbert':
            chexbert_all, chexbert_5 = CheXbert(refs_filename=base.format('refs.chexbert.txt'),
                                                hyps_filename=base.format('hyps.chexbert.txt'))(hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
        elif metric == 'radentitymatchexact':
            scores["radentitymatchexact"], _, _, _ = RadEntityMatchExact()(refs, hyps)
        elif metric == 'radentitynli':
            scores["radentitynli"], _ = RadEntityNLI()(refs, hyps)
        elif metric == 'radgraph':
            scores["radgraph"], _ = RadGraph()(refs=refs, hyps=hyps)
        else:
            logger.warning("Metric not implemented: {}".format(metric))

    with open(metrics_file, 'a+') as f:
        f.write(json.dumps({
            'split': split,
            'epoch': epoch,
            'scores': scores
        }, indent=4, sort_keys=False))
    return scores
