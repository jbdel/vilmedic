import torch
import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params

from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.huggingface.encoder.encoder_model import EncoderModel

from tqdm import tqdm
import numpy as np


def evaluation2(models, opts, dl):
    losses = []
    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:
        batch = {k: v.cuda() for k, v in batch.items()}
        results = [model(**batch) for model in models]
        losses.append(np.mean([r['loss'].mean().cpu().item() for r in results]))

    loss = np.mean(np.array(losses))

    return {'loss': loss}


def evaluation(models, opts, dl):
    losses = []
    linguistics = []
    visuals = []
    model = models[0]
    pbar = tqdm(dl, total=len(dl))
    for i, batch in enumerate(pbar):
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        out['loss'] = out['loss'].mean()
        losses.append(out['loss'].cpu().data.numpy())
        linguistics.append(out['linguistic'].cpu().data.numpy())
        visuals.append(out['visual'].cpu().data.numpy())

    linguistics = np.concatenate(linguistics, axis=0)
    visuals = np.concatenate(visuals, axis=0)
    split = dl.dataset.split
    np.save("{}_{}_visual".format(opts.file_name, split), visuals)
    np.save("{}_{}_linguistic".format(opts.file_name, split), linguistics)
    return {'loss': np.ndarray.mean(np.array(losses))}


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

    def __init__(self, encoder, cnn, projection, loss, **kwargs):
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

        loss, loss_l, loss_v = self.loss_fn(linguistic, visual)

        return {"loss": loss, "loss_l": loss_l, "loss_v": loss_v, "linguistic": linguistic, "visual": visual}

    # Necessary for generation
    def encoder(self, images, **kwargs):
        return self.enc(images)

    def __repr__(self):
        s = "ConVIRT\n"
        s += str(self.visual) + '\n'
        s += str(self.linguistic) + '\n'
        s += str(self.loss_fn) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
