import torch
import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
from pytorch_metric_learning.losses import NTXentLoss

from vilmedic.networks.blocks.vision import *

from tqdm import tqdm
import numpy as np


def evaluation(models, opts, dl, from_training, **kwargs):
    losses = []
    model = models[0]
    visuals = []

    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:
        out = model(batch, from_training=from_training)
        losses.append(out['loss'].mean().cpu().data.numpy())
        visuals.append(out['visual'].cpu().data.numpy())

    visuals = np.concatenate(visuals, axis=0)
    split = dl.dataset.split
    np.save("simclr_wik_{}_visual".format(split), visuals)
    return {'loss': np.ndarray.mean(np.array(losses))}


class SimCLR(nn.Module):

    def __init__(self, cnn, projection, loss, **kwargs):
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

        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, from_training, **kwargs):
        if not from_training:  # During test-time, we just return images embeddings
            return {"loss": torch.tensor(0.), "visual": self.vis_proj(self.visual(images.cuda()))}

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda()
        visual = self.vis_proj(self.visual(images))

        batch_size = visual.shape[0] // 2
        positive_embeddings, anchor_embeddings = torch.split(visual, [batch_size, batch_size], dim=0)
        embeddings = torch.cat((anchor_embeddings, positive_embeddings))
        indices = torch.arange(0, anchor_embeddings.size(0), device=anchor_embeddings.device)
        labels = torch.cat((indices, indices))

        loss = self.loss_fn(embeddings, labels)
        return {"loss": loss, "visual": visual}

    def __repr__(self):
        s = "SimCLR\n"
        s += str(self.visual) + '\n'
        s += str(self.loss_fn) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
