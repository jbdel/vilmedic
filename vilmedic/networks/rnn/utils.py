import numpy as np
import torch
import os


def readable_size(n):
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return "# parameters: {} ({} learnable)".format(
        readable_size(n_param_all), readable_size(n_param_learnable))


def set_embeddings(path, obj):
    filename = os.path.basename(path)
    embs = np.load(path, allow_pickle=True)
    assert len(embs) == obj.weight.size(0)
    success = 0
    with torch.no_grad():
        for i, emb in enumerate(embs):
            if emb is None:
                continue
            obj.weight.data[i] = torch.from_numpy(emb)
            success += 1
    print(filename, success, '/', len(embs), 'words loaded')
