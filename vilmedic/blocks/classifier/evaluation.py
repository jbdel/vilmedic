import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, config, dl, **kwargs):
    logits = np.array([])
    labels = np.array([])
    losses = np.array([])
    attentions = None
    cumulative_index = 0
    post_processing = {}

    for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
        label = batch['labels']
        batch_size = label.shape[0]
        num_classes = label.shape[1]

        batch = {k: v.cuda() if (isinstance(v, torch.Tensor) and torch.cuda.is_available()) else v for k, v in
                 batch.items()}
        results = [model(**batch) for model in models]

        # Pre-allocate memory
        if num_batch == 0:
            logits = np.zeros((len(dl.dataset), len(models), num_classes))
            labels = np.zeros((len(dl.dataset), num_classes))
            losses = np.zeros((len(dl), len(models)))
            if 'attentions' in results[0]:
                # results[0]['attentions'] is num_layers x batch_size x num_heads x sequence_length x sequence_length
                num_layers = len(results[0]['attentions'])
                attention_shape = results[0]['attentions'][0][0].shape
                attentions = np.zeros((len(dl.dataset), len(models), num_layers, *attention_shape))

        # iterating over the batch, stacking refs and hyps
        for i in range(batch_size):
            for j, r in enumerate(results):
                logits[cumulative_index + i][j] = r['output'][i].data.cpu().numpy()
            labels[cumulative_index + i] = label[i].data.cpu().numpy()

        # Loss
        for j, r in enumerate(results):
            losses[num_batch][j] = r['loss'].cpu().item()

        # Handle attention weights
        if attentions is not None:
            for j, r in enumerate(results):
                for k, attention_layer in enumerate(r['attentions']):
                    for i in range(batch_size):
                        attentions[cumulative_index + i][j][k] = attention_layer[i].data.cpu().numpy()

        cumulative_index += batch_size

    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)

    if attentions:
        post_processing["attentions"] = np.array(attentions)

    return {**{'loss': loss, 'refs': labels, 'hyps': preds, 'logits': logits}, **post_processing}
