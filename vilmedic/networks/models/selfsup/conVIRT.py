import torch
import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params

from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.huggingface.encoder.encoder_model import EncoderModel

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


class ConVIRTLoss(nn.Module):
    def __init__(self, tau, lambda_, **kwargs):
        super().__init__()
        self.tau = tau
        self.lambda_ = lambda_
        self.cos_loss = nn.CosineSimilarity()

    def forward(self, linguistic, visual):
        nominator = torch.exp(torch.div(self.cos_loss(linguistic, visual), self.tau))

        denominator_l = self.pairwise_cosine_distance(linguistic, visual)
        denominator_l = torch.sum(torch.exp(torch.div(denominator_l, self.tau)), dim=1)
        loss_l = -torch.log(torch.div(nominator, denominator_l))

        denominator_v = self.pairwise_cosine_distance(visual, linguistic)
        denominator_v = torch.sum(torch.exp(torch.div(denominator_v, self.tau)), dim=1)
        loss_v = -torch.log(torch.div(nominator, denominator_v))
        loss = torch.mean(self.lambda_ * loss_v + (1 - self.lambda_) * loss_l)
        return loss, loss_l, loss_v

    @staticmethod
    def pairwise_cosine_distance(a, b):
        eps = 1e-08
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def __repr__(self):
        return "ConVIRTLoss(\n" + \
               "\t(cos_loss): CosineSimilarity()\n" + \
               "\t(tau): {}\n".format(self.tau) + \
               "\t(lambda_): {}\n".format(self.lambda_) + \
               ")"


class InfoNCELoss(nn.Module):
    def __init__(self, tau, **kwargs):
        super().__init__()
        self.tau = tau
        self.xe = nn.CrossEntropyLoss(reduction='none')

    def forward(self, linguistic, visual):
        n = linguistic.shape[0]
        logits = linguistic @ visual.T
        labels = torch.arange(n).cuda()
        loss_t = self.xe(logits, labels)
        loss_i = self.xe(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        loss = loss.mean()
        return loss, loss_t, loss_i

    def __repr__(self):
        return "InfoNCELoss(\n" + \
               "\t(tau): {}\n".format(self.tau) + \
               ")"


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
        for i in range(int(bs / min(self.fbs, bs))):
            inp = input_ids[i * self.fbs:(i + 1) * self.fbs]
            att = attention_mask[i * self.fbs:(i + 1) * self.fbs]
            im = images[i * self.fbs:(i + 1) * self.fbs]

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
