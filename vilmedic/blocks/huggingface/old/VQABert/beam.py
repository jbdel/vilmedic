# from . import seq2seq
# from .beam_helpers import generate
from ..beam_helpers import generate
import tqdm
import torch
import os
import collections

def beam_search(models, config, dl):
    ref_list = []
    hyp_list = collections.defaultdict(list)

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            ref_list.extend(batch['labels'].data.cpu().numpy())
            for i, m in enumerate(models):
                logits = m(**batch)["logits"]
                hyp_list[i].extend(logits)

        hyp_list = torch.stack([torch.stack(v) for v in hyp_list.values()])
        hyp_list = hyp_list.mean(0)
        hyp_list = torch.argmax(hyp_list, dim=-1).data.cpu().numpy()

        answers = [dl.dataset.labels_dict.idx2label(h) for h in hyp_list]

        f = open(os.path.join(dl.dataset.ckpt_dir, dl.dataset.split + '_answers'), 'w')
        f.write('\n'.join(answers))
        return 0.0, ref_list, hyp_list
