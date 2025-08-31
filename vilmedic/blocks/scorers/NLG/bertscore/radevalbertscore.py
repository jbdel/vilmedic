import torch
from bert_score import score

def _get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RadEvalBERTScorer:
    """
    Wrapper around bert_score for radiology reports using a custom BERT model.
    """
    def __init__(self,
                 model_type: str = "IAMJB/RadEvalModernBERT",
                 num_layers: int = None,
                 use_fast_tokenizer: bool = True,
                 rescale_with_baseline: bool = False,
                 device: torch.device = None):
        self.model_type = model_type
        self.num_layers = num_layers
        self.use_fast_tokenizer = use_fast_tokenizer
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device or _get_default_device()

    def score(self, refs: list[str], hyps: list[str]) -> float:
        """
        Compute BERTScore F1 between reference and hypothesis texts.

        Args:
            refs: list of reference sentences.
            hyps: list of hypothesis sentences (predictions).

        Returns:
            Mean F1 score as a float.
        """
        # bert_score expects cands (hypotheses) first, then refs
        P, R, F1 = score(
            cands=hyps,
            refs=refs,
            model_type=self.model_type,
            num_layers=self.num_layers,
            use_fast_tokenizer=self.use_fast_tokenizer,
            rescale_with_baseline=self.rescale_with_baseline,
            device=self.device
        )
        # Return the mean F1 over all pairs
        return F1.mean().item()

if __name__ == "__main__":
    # Example usage
    refs = ["Chronic mild to moderate cardiomegaly and pulmonary venous hypertension."]
    hyps = ["Mild left basal atelectasis; no pneumonia."]
    scorer = RadiologyBERTScorer(num_layers=23)
    f1_score = scorer.score(refs, hyps)
    print(f"Mean F1 score: {f1_score:.4f}")