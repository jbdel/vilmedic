import json
import os
import torch

from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from dalle_pytorch import CLIP as pyCLIP
from torch import nn
import torch.nn.functional as F

from vilmedic.networks.models.utils import get_n_params


def evaluation(models, config, dl, **kwargs):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    losses = []
    linguistics = []
    visuals = []
    linguistics_norm = []
    visuals_norm = []

    model = models[0]
    pbar = tqdm(dl, total=len(dl))

    if isinstance(model, torch.nn.DataParallel):
        model.module.clip.to_text_latent.register_forward_hook(get_activation('to_text_latent'))
        model.module.clip.to_visual_latent.register_forward_hook(get_activation('to_visual_latent'))
    else:
        model.clip.to_text_latent.register_forward_hook(get_activation('to_text_latent'))
        model.clip.to_visual_latent.register_forward_hook(get_activation('to_visual_latent'))

    for batch in pbar:
        out = model(**batch)
        out['loss'] = out['loss'].mean()
        losses.append(out['loss'].cpu().data.numpy())

        text_latents = activation['to_text_latent']
        image_latents = activation['to_visual_latent']

        linguistics.append(text_latents.cpu().data.numpy())
        visuals.append(image_latents.cpu().data.numpy())

        text_latents_norm, image_latents_norm = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, image_latents))

        linguistics_norm.append(text_latents_norm.cpu().data.numpy())
        visuals_norm.append(image_latents_norm.cpu().data.numpy())

    linguistics = np.concatenate(linguistics, axis=0)
    visuals = np.concatenate(visuals, axis=0)
    linguistics_norm = np.concatenate(linguistics_norm, axis=0)
    visuals_norm = np.concatenate(visuals_norm, axis=0)
    split = dl.dataset.split
    np.save("CLIP_mimic_{}_visual".format(split), visuals)
    np.save("CLIP_mimic_{}_linguistic".format(split), linguistics)
    np.save("CLIP_norm_mimic_{}_visual".format(split), visuals_norm)
    np.save("CLIP_norm_mimic_{}_linguistic".format(split), linguistics_norm)

    return {'loss': np.ndarray.mean(np.array(losses))}


class CLIP(nn.Module):
    def __init__(self,
                 clip,
                 **kwargs):
        super(CLIP, self).__init__()
        self.clip_config = clip
        self.clip = pyCLIP(
            **self.clip_config
        )

        self.eval_func = evaluation

    def forward(self, input_ids, images, **kwargs):
        loss = self.clip(input_ids.cuda(), images.cuda(), text_mask=torch.ones_like(input_ids).bool().cuda(),
                         return_loss=True)
        return {'loss': loss}

    def __repr__(self):
        s = str(type(self.clip).__name__) + '(' + str(
            json.dumps(dict(self.clip_config), indent=4, sort_keys=True)) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
