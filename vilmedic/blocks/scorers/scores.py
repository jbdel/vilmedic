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
from .utils import get_logger_directory
from omegaconf import OmegaConf

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
    "chexbert": [F1CheXbert, 1],
    "radgraph": [F1RadGraph, 1],
    "bertscore": [BertScore, 1],
}


def compute_scores(metrics, refs, hyps, split, seed, config, epoch, logger, dump=True):
    """Compute evaluation metrics for generated hypotheses against references."""
    scores = dict()
    
    # Early return if no metrics specified
    if metrics is None or not metrics:
        return scores

    # Validate inputs
    assert refs is not None and hyps is not None, \
        "You specified metrics but your evaluation does not return hyps nor refs"
    assert len(refs) == len(hyps), \
        f'refs and hyps must have same length: {len(refs)} vs {len(hyps)}'

    # Dump predictions if requested
    if dump:
        base = os.path.join(get_logger_directory(logger), f'{split}_{seed}_{{}}')
        refs_file = base.format('refs.txt')
        hyps_file = base.format('hyps.txt')
        metrics_file = base.format('metrics.txt')

        with open(refs_file, 'w') as f:
            f.write('\n'.join(map(str, refs)))

        with open(hyps_file, 'w') as f:
            f.write('\n'.join(map(str, hyps)))

    # Process each metric
    for metric in metrics:
        metric_args = dict()
        metric_name = metric
        
        # Handle metrics with arguments (dict format)
        if OmegaConf.is_dict(metric) or isinstance(metric, dict):
            if len(metric) != 1:
                logger.warning(f"Metric badly formatted: {metric}. Expecting single key-value pair.")
                continue
            metric_key = list(metric.keys())[0]
            metric_args = metric[metric_key]
            metric_name = metric_key
        
        # For simple string metrics, metric_name is already set
        # Normalize metric name to lowercase for comparison
        metric_lower = str(metric_name).lower()

        # Compute the appropriate metric
        try:
            if metric_lower == 'bleu':
                scores["BLEU"] = Bleu()(refs, hyps)[0]
                
            elif metric_lower == 'meteor':
                scores["METEOR"] = Meteor()(refs, hyps)[0]
                
            elif metric_lower == 'ciderd':
                scores["CIDERD"] = CiderD()(refs, hyps)[0]
                
            elif metric_lower == 'bertscore':
                scores["bertscore"] = BertScore()(refs, hyps)[0]
                
            elif metric_lower == 'radevalbertscore':
                scores["radevalbertscore"] = RadEvalBERTScorer(
                    model_type="IAMJB/RadEvalModernBERT", 
                    num_layers=22,
                    use_fast_tokenizer=True,
                    rescale_with_baseline=False
                ).score(refs, hyps)
                
            elif metric_lower in ['rouge1', 'rouge2', 'rougel']:
                scores[metric_name.upper()] = Rouge(rouges=[metric_lower])(refs, hyps)[0]
                
            elif metric_lower == 'accuracy':
                scores["accuracy"] = round(np.mean(np.array(refs) == np.argmax(hyps, axis=-1)) * 100, 2)
                
            elif metric_lower == 'f1-score':
                scores["f1-score"] = classification_report(refs, np.argmax(hyps, axis=-1))
                
            elif metric_lower == 'auroc':
                scores["auroc"] = roc_auc_score(
                    refs, 
                    F.softmax(torch.from_numpy(hyps), dim=-1).numpy(), 
                    multi_class="ovr"
                )
                
            elif metric_lower == 'chexbert':
                accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = F1CheXbert(
                    refs_filename=base.format('refs.chexbert.txt') if dump else None,
                    hyps_filename=base.format('hyps.chexbert.txt') if dump else None
                )(hyps, refs)
                scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
                scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
                scores["chexbert-5_macro avg_f1-score"] = chexbert_5["macro avg"]["f1-score"]
                scores["chexbert-all_macro avg_f1-score"] = chexbert_all["macro avg"]["f1-score"]
                
            elif metric_lower == 'radentitymatchexact':
                scores["radentitymatchexact"] = RadEntityMatchExact()(refs, hyps)[0]
                
            elif metric_lower == 'radentitynli':
                scores["radentitynli"] = RadEntityNLI()(refs, hyps)[0]
                
            elif metric_lower == 'radgraph':
                results = F1RadGraph(reward_level="all", model_type="radgraph-xl")(refs=refs, hyps=hyps)[0]
                scores["radgraph_simple"], scores["radgraph_partial"], scores["radgraph_complete"] = results
                
            elif metric_lower == 'stanford_ct_abd_accuracy':
                scores["stanford_ct_abd"] = StanfordCTAbdAcc()(refs=refs, hyps=hyps)[0]
                
            else:
                logger.warning(f"Metric not implemented: {metric_name}")
                
        except Exception as e:
            logger.error(f"Error computing metric {metric_name}: {str(e)}")
            scores[metric_name] = None

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
