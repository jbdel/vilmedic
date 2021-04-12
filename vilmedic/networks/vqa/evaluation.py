import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, opts, dl):
    # labels_dir = Labels(opts.label_index)

    logits = []
    labels = []
    losses = []
    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:

        label = batch['label']
        batch_size = len(label)

        batch = {k: v.cuda() for k, v in batch.items()}
        results = [model(**batch, **opts) for model in models]

        # iterating over the batch, stacking logits
        for i in range(batch_size):
            logits.append([r['output'][i].data.cpu().numpy() for r in results])
            labels.append(label[i].data.cpu().numpy())

        # getting average loss
        losses.append(np.mean([r['loss'].cpu().item() for r in results]))

    logits_avg = np.mean(np.array(logits), axis=1)
    preds = np.argmax(np.array(logits_avg), axis=-1)

    return np.array(losses), np.array(labels), np.array(preds)
