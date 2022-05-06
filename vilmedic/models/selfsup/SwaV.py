import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params
import json
from pl_bolts.models.self_supervised import SwAV
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
import copy


def sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        K, B = Q.shape

        u = torch.zeros(K).cuda()
        r = torch.ones(K).cuda() / K
        c = torch.ones(B).cuda() / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def evaluation(models, config, dl, from_training, **kwargs):
    # No ensembling for this evaluation
    model = models[0]

    losses = []
    visuals = []

    pbar = tqdm(dl, total=len(dl))
    for i, batch in enumerate(pbar):
        out = model(**batch, from_training=from_training)
        losses.append(out['loss'].mean().cpu().data.numpy())

        if not from_training:
            visuals.append(out['visual'].cpu())
        # break

    if from_training:
        return {'loss': np.ndarray.mean(np.array(losses))}

    return {'loss': np.ndarray.mean(np.array(losses)),
            'visual': torch.cat(visuals)
            }


class SwaV(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = copy.deepcopy(config)

        # Visual Encoder
        self.sinkhorn_iterations = config.pop('sinkhorn_iterations')
        self.hidden_mlp = config.pop('hidden_mlp')
        self.output_dim = config.pop('output_dim')
        self.nmb_prototypes = config.pop('nmb_prototypes')
        self.first_conv = config.pop('first_conv')
        self.maxpool1 = config.pop('maxpool1')

        self.crops_for_assign = config.pop('crops_for_assign')
        self.nmb_crops = config.pop('nmb_crops')
        self.temperature = config.pop('temperature')
        self.epsilon = config.pop('epsilon')

        arch = config.arch
        self.model = eval(arch)(
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.output_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
        )

        self.softmax = nn.Softmax(dim=1)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, from_training=True, **kwargs):
        if not from_training:
            return {"loss": torch.tensor(0.),
                    'visual': self.model.projection_head(self.model.forward_backbone(images.cuda()))}

        # https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
        # 1. normalize the prototypes
        images = images[:-1]
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(images)
        embedding = embedding.detach()
        bs = images[0].size(0)

        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # 4. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = sinkhorn(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return {"loss": loss}

    def __repr__(self):
        s = "SwaV\n"
        s += str(json.dumps(OmegaConf.to_container(self.config), indent=4)) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
