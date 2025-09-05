import os
import numpy as np
import logging
from transformers import EvalPrediction
from vilmedic.blocks.scorers.NLG.bertscore.bertscore import BertScore
from vilmedic.blocks.scorers.NLG.bertscore.radevalbertscore import RadEvalBERTScorer


def compute_metrics_factory(metrics_list, tokenizer, save_dir, logger, seed=None, is_main_process=True, context=None):
    """Factory function to create compute_metrics function for Trainer, also saves decoded preds/refs.
    
    Args:
        metrics_list: List of metric names to compute
        tokenizer: Tokenizer for decoding predictions
        save_dir: Directory to save predictions/references
        logger: Logger instance
        seed: Training seed to include in filename
    """
    # eval counter for file naming
    eval_counter = {"n": 1}
    
    def compute_metrics(eval_pred: EvalPrediction):
        # Hard guard: skip entirely on non-main processes
        if not is_main_process:
            return {}
        """Compute metrics for evaluation"""
        # Increase eval counter
        eval_counter["n"] += 1
        cur_eval_idx = eval_counter["n"]
        
        # Get predictions and labels
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # For generation tasks, predictions are usually the generated token ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Skip if predictions are None or empty
        if predictions is None or len(predictions) == 0:
            logger.warning("[Metrics] No predictions to evaluate")
            return {}
        
        # Decode predictions
        if predictions.dtype == np.float32 or predictions.dtype == np.float64:
            # If predictions are logits, take argmax
            predictions = np.argmax(predictions, axis=-1)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Handle labels - replace -100 with pad token
        if labels is not None:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            decoded_labels = [""] * len(decoded_preds)
        
        # Clean up texts
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Log evaluation info
        logger.info(f"[Metrics] Epoch {cur_eval_idx}: Processing {len(decoded_preds)} predictions")
        
        # Save decoded preds/refs with seed in filename (main process only)
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Optional split tag in filenames
            split_tag = None
            if context and isinstance(context, dict):
                split_tag = context.get('split', None)

            # Add seed to prediction filename if provided
            if seed:
                pred_filename = f"preds_epoch{cur_eval_idx}_seed{seed}.txt"
            else:
                pred_filename = f"preds_epoch{cur_eval_idx}.txt"

            # Append split tag if present
            if split_tag:
                base, ext = os.path.splitext(pred_filename)
                pred_filename = f"{base}_{split_tag}{ext}"

            # References don't need seed as they're always the same
            ref_filename = f"refs_epoch{cur_eval_idx}.txt"
            if split_tag:
                base, ext = os.path.splitext(ref_filename)
                ref_filename = f"{base}_{split_tag}{ext}"

            pred_path = os.path.join(save_dir, pred_filename)
            ref_path = os.path.join(save_dir, ref_filename)

            with open(pred_path, "w") as f:
                f.write("\n".join(map(str, decoded_preds)))
            with open(ref_path, "w") as f:
                f.write("\n".join(map(str, decoded_labels)))

            logger.info(f"[Metrics] Saved predictions to {pred_filename}")
            logger.info(f"[Metrics] Saved references to {ref_filename}")

        except Exception as e:
            logger.error(f"[Metrics] Could not save decoded predictions: {e}")
        
        # Compute metrics
        results = {}
        logger.info(f"[Metrics] Computing {len(metrics_list)} metrics...")

        # Dispatch table for metrics
        def _compute_bertscore():
            score = BertScore()(decoded_labels, decoded_preds)[0]
            return 'bertscore', score

        def _compute_radevalbertscore():
            scorer = RadEvalBERTScorer(
                model_type="IAMJB/RadEvalModernBERT",
                num_layers=22,
                use_fast_tokenizer=True,
                rescale_with_baseline=False
            )
            score = scorer.score(decoded_labels, decoded_preds)
            return 'radevalbertscore', score

        metric_dispatch = {
            'bertscore': _compute_bertscore,
            'radevalbertscore': _compute_radevalbertscore,
        }

        for metric_name in metrics_list:
            metric_lower = metric_name.lower()
            handler = metric_dispatch.get(metric_lower)

            if handler:
                try:
                    name, score = handler()
                    results[name] = score
                    logger.info(f"[Metrics] {name}: {score:.4f}")
                except Exception as e:
                    logger.error(f"[Metrics] Error computing {metric_lower}: {e}")
                    results[metric_lower] = 0.0
            else:
                logger.warning(f"[Metrics] Metric {metric_name} not implemented")
        
        logger.info(f"[Metrics] Epoch {cur_eval_idx} evaluation complete")
        return results
    
    return compute_metrics