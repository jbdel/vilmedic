import torch.nn as nn
import math
from tqdm import tqdm
import numpy as np
from dalle_pytorch import DiscreteVAE

from vilmedic.networks.models.utils import get_n_params


def evaluation(models, config, dl, **kwargs):
    losses = []
    model = models[0]
    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:
        out = model(**batch)
        out['loss'] = out['loss'].mean()
        losses.append(out['loss'].cpu().data.numpy())
    return {'loss': np.ndarray.mean(np.array(losses))}


class VAE(nn.Module):
    def __init__(self,
                 image_size=256,
                 num_layers=3,
                 num_tokens=8192,
                 codebook_dim=512,
                 hidden_dim=64,
                 num_resnet_blocks=1,
                 temperature=0.9,
                 straight_through=False,
                 **kwargs):
        super(VAE, self).__init__()

        self.vae = DiscreteVAE(
            image_size=image_size,
            num_layers=num_layers,
            num_tokens=num_tokens,
            codebook_dim=codebook_dim,
            hidden_dim=hidden_dim,
            num_resnet_blocks=num_resnet_blocks,
            temperature=temperature,
            straight_through=straight_through
        )

        self.global_step = 0
        self.temp = temperature

        self.eval_func = evaluation

    def forward(self, images, **kwargs):
        loss, output = self.vae(images.cuda(),
                                return_loss=True,
                                return_recons=True,
                                temp=self.temp
                                )
        self.global_step += 1
        if self.global_step % 100 == 0:
            self.temp = max(self.temp * math.exp(-1e-6 * self.global_step), 0.5)

        return {'loss': loss, 'output': output}

    def __repr__(self):
        s = "DiscreteVAE()" + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
