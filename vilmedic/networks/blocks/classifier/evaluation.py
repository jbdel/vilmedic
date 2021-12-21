import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, config, dl, **kwargs):
    logits = []
    labels = []
    losses = []
    attentions = []

    post_processing = {}

    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:

        label = batch['labels']
        batch_size = len(label)

        batch = {k: v.cuda() for k, v in batch.items()}
        results = [model(**batch) for model in models]
        # iterating over the batch, stacking refs and hyps
        for i in range(batch_size):
            logits.append([r['output'][i].data.cpu().numpy() for r in results])
            labels.append(label[i].data.cpu().numpy())

        # Do we have attention weights?
        if 'attentions' in results[0]:
            # attentions will be of size num_models x bs x num_layers x num_heads x seq_len x seq_len
            for i in range(batch_size):
                attentions.append([torch.stack(list(r['attentions']))[:, i, :].data.cpu().numpy() for r in results])

        # getting average loss
        losses.append(np.mean([r['loss'].cpu().item() for r in results]))

    preds = np.mean(np.array(logits), axis=1)
    loss = np.mean(np.array(losses))

    if attentions:
        post_processing["attentions"] = np.array(attentions)

    return {**{'loss': loss, 'refs': np.array(labels), 'hyps': np.array(preds)}, **post_processing}
