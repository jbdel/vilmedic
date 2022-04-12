import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.vision import CNN
from vilmedic.blocks.losses import VICREGLoss

from vilmedic.blocks.huggingface.encoder.encoder_model import EncoderModel

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
            linguistics.append(out['linguistic'].cpu())
            visuals.append(out['visual'].cpu())

    if from_training:
        return {'loss': np.ndarray.mean(np.array(losses))}

    return {'loss': np.ndarray.mean(np.array(losses)),
            'linguistic': torch.cat(linguistics),
            'visual': torch.cat(visuals)
            }


class VICREG(nn.Module):

    def __init__(self, encoder, cnn, projection, loss, **kwargs):
        super().__init__()

        # Linguistic encoder
        self.linguistic = EncoderModel(encoder)

        # Visual Encoder
        self.visual = eval(cnn.pop('proto'))(**cnn)

        # Projection
        self.vis_proj = nn.Sequential(
            nn.Linear(projection.visual_embedding_dim, projection.proj_hidden_dim),
            nn.BatchNorm1d(projection.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection.proj_hidden_dim, projection.proj_hidden_dim),
            nn.BatchNorm1d(projection.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection.proj_hidden_dim, projection.proj_output_dim),
        )

        self.lin_proj = nn.Sequential(
            nn.Linear(projection.textual_embedding_dim, projection.proj_hidden_dim),
            nn.BatchNorm1d(projection.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection.proj_hidden_dim, projection.proj_hidden_dim),
            nn.BatchNorm1d(projection.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection.proj_hidden_dim, projection.proj_output_dim),
        )

        self.loss_fn = VICREGLoss(**loss)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, **kwargs):
        images = images.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        linguistic = self.linguistic(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     **kwargs)
        linguistic = self.lin_proj(linguistic['pooler_output'])
        visual = self.vis_proj(self.visual(images))

        vicreg_loss = self.loss_fn(linguistic, visual)

        return {"loss": vicreg_loss, "linguistic": linguistic, "visual": visual}

    def __repr__(self):
        s = "VICREG\n"
        s += str(self.visual) + '\n'
        s += str(self.linguistic) + '\n'
        s += str(self.lin_proj) + '\n'
        s += str(self.loss_fn) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
