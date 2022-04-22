from mauve.utils import get_model, get_tokenizer, featurize_tokens_from_model
from mauve.compute_mauve import cluster_feats, get_divergence_curve_for_multinomials, compute_area_under_curve, \
    get_fronter_integral, SimpleNamespace
import numpy as np
import torch.nn as nn

def compute_mauve(
        p_features, q_features,
        num_buckets='auto', pca_max_data=-1, kmeans_explained_var=0.9,
        kmeans_num_redo=5, kmeans_max_iter=500,
        divergence_curve_discretization_size=25, mauve_scaling_factor=5,
        verbose=False, seed=25
):
    if num_buckets == 'auto':
        # heuristic: use num_clusters = num_generations / 10
        num_buckets = max(2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10)))
    elif not isinstance(num_buckets, int):
        raise ValueError('num_buckets is expected to be an integer or "auto"')

    # Acutal binning
    p, q = cluster_feats(p_features, q_features,
                         num_clusters=num_buckets,
                         norm='l2', whiten=False,
                         pca_max_data=pca_max_data,
                         explained_variance=kmeans_explained_var,
                         num_redo=kmeans_num_redo,
                         max_iter=kmeans_max_iter,
                         seed=seed, verbose=verbose)
    # Divergence curve and mauve
    mixture_weights = np.linspace(1e-6, 1 - 1e-6, divergence_curve_discretization_size)
    divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, mauve_scaling_factor)
    x, y = divergence_curve.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)
    mauve_score = 0.5 * (
            compute_area_under_curve(x[idxs1], y[idxs1]) +
            compute_area_under_curve(y[idxs2], x[idxs2])
    )
    fi_score = get_fronter_integral(p, q)
    to_return = SimpleNamespace(
        p_hist=p, q_hist=q, divergence_curve=divergence_curve,
        mauve=mauve_score,
        frontier_integral=fi_score,
        num_buckets=num_buckets,
    )
    return to_return


class MauveScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = get_tokenizer(self.model)
        self.model = get_model(self.model, self.tokenizer, -1)

    def forward(self, refs, hyps):
        tokenized_refs = [self.tokenizer.encode(sen, return_tensors='pt', truncation=True, max_length=512)
                          for sen in refs]
        tokenized_hyps = [self.tokenizer.encode(sen, return_tensors='pt', truncation=True, max_length=512)
                          for sen in hyps]
        p_features = featurize_tokens_from_model(self.model, tokenized_refs, name="p").detach().cpu().numpy()
        q_features = featurize_tokens_from_model(self.model, tokenized_hyps, name="q").detach().cpu().numpy()

        return compute_mauve(p_features=p_features, q_features=q_features).mauve


if __name__ == '__main__':
    m = MauveScorer(model=None or "distilgpt2")
    refs = [
        'no evidence of consolidation to suggest pneumonia is seen. there  is some retrocardiac atelectasis. a small left pleural effusion may be  present. no pneumothorax is seen. no pulmonary edema. a right granuloma is  unchanged. the heart is mildly enlarged, unchanged. there is tortuosity of  the aorta.',
        'there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.'
    ]
    hyps = [
        'heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
        'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.'
    ]
    print(m.compute(refs, hyps))
