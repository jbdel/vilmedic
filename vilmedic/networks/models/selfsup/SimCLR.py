import torch
import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
from pytorch_metric_learning.losses import NTXentLoss

from vilmedic.networks.blocks.vision import *

from tqdm import tqdm
import numpy as np


def evaluation(models, config, dl, from_training, **kwargs):
    # No ensembling for this evaluation
    model = models[0]
    losses = []
    visuals = []

    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:
        out = model(**batch, from_training=from_training)
        losses.append(out['loss'].mean().cpu().data.numpy())
        if not from_training:
            visuals.append(out['visual'].cpu().data)

    if from_training:
        return {'loss': np.ndarray.mean(np.array(losses))}

    return {'loss': np.ndarray.mean(np.array(losses)),
            'visual': torch.cat(visuals)
            }


class SimCLR(nn.Module):

    def __init__(self, cnn, projection, loss, forward_batch_size=256, **kwargs):
        super().__init__()

        # Visual Encoder
        self.visual = eval(cnn.pop('proto'))(**cnn)

        # Projection
        self.vis_proj = nn.Sequential(
            nn.Linear(projection.visual_embedding_dim, projection.projection_dim),
            nn.ReLU(),
            nn.Linear(projection.projection_dim, projection.projection_dim),
        )

        self.loss_fn = NTXentLoss(temperature=loss.tau)
        self.fbs = forward_batch_size

        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, from_training=True, **kwargs):
        if from_training:
            images = torch.cat(torch.split(images, [3, 3], dim=1), dim=0)

        # forward passes
        visuals = []
        images = images.cuda()
        bs = images.shape[0]
        for i in range(int(bs / min(self.fbs, bs))):
            im = images[i * self.fbs:(i + 1) * self.fbs]
            visual = self.vis_proj(self.visual(im))
            visuals.append(visual)
        visuals = torch.cat(visuals)

        # if from ensemblor, return embeddings
        if not from_training:
            return {"loss": torch.tensor(0.), "visual": visuals}

        # Computing loss
        batch_size = visuals.shape[0] // 2
        indices = torch.arange(0, batch_size, device=visuals.device)
        labels = torch.cat((indices, indices))
        loss = self.loss_fn(visuals, labels)

        return {"loss": loss}

    def __repr__(self):
        s = "SimCLR\n"
        s += str(self.visual) + '\n'
        s += str(self.loss_fn) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
