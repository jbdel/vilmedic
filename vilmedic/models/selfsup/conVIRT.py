import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.encoder.encoder_model import EncoderModel
from vilmedic.blocks.losses import ConVIRTLoss, InfoNCELoss

from tqdm import tqdm
import numpy as np


def evaluation(models, config, dl, from_training, **kwargs):
    # No ensembling for this evaluation
    model = models[0]

    losses = []
    linguistics = []
    visuals = []

    pbar = tqdm(dl, total=len(dl))
    for i, batch in enumerate(pbar):
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        losses.append(out['loss'].mean().cpu().data.numpy())

        if not from_training:
            linguistics.append(out['linguistic'].cpu().data)
            visuals.append(out['visual'].cpu().data)

    if from_training:
        return {'loss': np.ndarray.mean(np.array(losses))}

    return {'loss': np.ndarray.mean(np.array(losses)),
            'linguistic': torch.cat(linguistics),
            'visual': torch.cat(visuals)
            }


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ConVIRT(nn.Module):

    def __init__(self, encoder, cnn, projection, loss, forward_batch_size=256, **kwargs):
        super().__init__()

        # Linguistic encoder
        self.linguistic = EncoderModel(encoder)

        # Visual Encoder
        self.visual = eval(cnn.pop('proto'))(**cnn)

        # Projection
        self.vis_proj = nn.Sequential(
            nn.Linear(projection.visual_embedding_dim, projection.projection_dim),
            nn.ReLU(),
            nn.Linear(projection.projection_dim, projection.projection_dim),
        )
        self.lin_proj = nn.Sequential(
            nn.Linear(projection.textual_embedding_dim, projection.projection_dim),
            nn.ReLU(),
            nn.Linear(projection.projection_dim, projection.projection_dim),
        )

        self.loss_fn = eval(loss.pop("proto"))(**loss)
        self.fbs = forward_batch_size

        # Evaluation
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, **kwargs):
        images = images.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        bs = images.shape[0]
        linguistics = []
        visuals = []
        for i in list(chunks(range(bs), min(self.fbs, bs))):
            inp = input_ids[i]
            att = attention_mask[i]
            im = images[i]

            linguistic = self.linguistic(input_ids=inp,
                                         attention_mask=att,
                                         **kwargs)
            linguistic = self.lin_proj(linguistic['pooler_output'])
            visual = self.vis_proj(self.visual(im))

            linguistics.append(linguistic)
            visuals.append(visual)

        linguistics = torch.cat(linguistics)
        visuals = torch.cat(visuals)

        loss, loss_l, loss_v = self.loss_fn(linguistics, visuals)

        return {"loss": loss, "loss_l": loss_l, "loss_v": loss_v, "linguistic": linguistics, "visual": visuals}

    def __repr__(self):
        s = "ConVIRT\n"
        s += str(self.visual) + '\n'
        s += str(self.linguistic) + '\n'
        s += str(self.loss_fn) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
