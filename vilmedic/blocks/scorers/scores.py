import os
import numpy as np
import json
import torch.nn.functional as F
import torch
import logging
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
from sklearn.metrics import classification_report, roc_auc_score
from . import *

# RadGraph package overrides logger, need to set back to default
logging.setLoggerClass(logging.Logger)

REWARD_COMPLIANT = {
    "rougel": [RougeL, 1],
    "rouge2": [Rouge2, 1],
    "rouge1": [Rouge1, 1],
    "bleu": [Bleu, 1],
    "meteor": [Meteor, 1],
    "ciderdrl": [CiderDRL, 1],
    "radentitymatchexact": [RadEntityMatchExact, 1],
    "radentitynli": [RadEntityNLI, 1],
    "chexbert": [CheXbert, 1],
    "radgraph": [F1RadGraph, 1],
    "bertscore": [BertScore, 1],
    # "MAUVE": [MauveScorer, 0],
}


def compute_scores(metrics, refs, hyps, split, seed, config, epoch, logger, dump=True):
    scores = dict()
    # If metric is None or empty list
    if metrics is None or not metrics:
        return scores

    assert refs is not None and hyps is not None, \
        "You specified metrics but your evaluation does not return hyps nor refs"

    assert len(refs) == len(hyps), 'refs and hyps must have same length : {} vs {}'.format(len(refs), len(hyps))

    # Dump
    if dump:
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
        # metric_args = dict()
        #
        # # if metric has arguments
        # if OmegaConf.is_dict(metric):
        #     if len(metric) != 1:
        #         logger.warning("Metric badly formatted: {}. Skipping.".format(metric))
        #         continue
        #     metric_args = metric[list(metric.keys())[0]]
        #     metric = list(metric.keys())[0]

        # Iterating over metrics
        if metric == 'BLEU':
            scores["BLEU"] = Bleu()(refs, hyps)[0]
        elif metric == 'METEOR':
            scores["METEOR"] = Meteor()(refs, hyps)[0]
        elif metric == 'CIDERD':
            scores["CIDERD"] = CiderD()(refs, hyps)[0]
        elif metric == 'bertscore':
            scores["bertscore"] = BertScore()(refs, hyps)[0]
        elif metric in ['ROUGE1', 'ROUGE2', 'ROUGEL']:
            scores[metric] = Rouge(rouges=[metric.lower()])(refs, hyps)[0]
        elif metric == 'MAUVE':
            scores["MAUVE"] = round(
                MauveScorer(config.mauve_featurize_model_name or "distilgpt2")(refs, hyps) * 100, 2)
        elif metric == 'accuracy':
            scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
        elif metric == 'f1-score':
            scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
        elif metric == 'auroc':
            scores["auroc"] = roc_auc_score(refs, F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), multi_class="ovr")
        elif metric == 'chexbert':
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = F1CheXbert(
                refs_filename=base.format('refs.chexbert.txt') if dump else None,
                hyps_filename=base.format('hyps.chexbert.txt') if dump else None) \
                (hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
            scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]
        elif metric == 'radentitymatchexact':
            scores["radentitymatchexact"] = RadEntityMatchExact()(refs, hyps)[0]
        elif metric == 'radentitynli':
            scores["radentitynli"] = RadEntityNLI()(refs, hyps)[0]
        elif metric == 'radgraph':
            scores["radgraph_simple"], scores["radgraph_partial"], scores["radgraph_complete"] = \
                F1RadGraph(reward_level="all")(refs=refs, hyps=hyps)[0]
        elif metric == 'stanford_ct_abd_accuracy':
            scores["stanford_ct_abd"] = StanfordCTAbdAcc()(refs=refs, hyps=hyps)[0]
        else:
            logger.warning("Metric not implemented: {}".format(metric))

    if dump:
        with open(metrics_file, 'a+') as f:
            f.write(json.dumps({
                'split': split,
                'epoch': epoch,
                'scores': scores
            }, indent=4, sort_keys=False))
    return scores


if __name__ == '__main__':
    pass
