import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, Union
from diffusers.models import VQModel as HFVQModel


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

    return {'loss': np.ndarray.mean(np.array(losses)),
            'visual': torch.cat(visuals)
            }


class VQModel(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 3,
                 sample_size: int = 32,
                 num_vq_embeddings: int = 256,
                 norm_num_groups: int = 32,
                 vq_embed_dim: Optional[int] = None,
                 scaling_factor: float = 0.18215,
                 norm_type: str = "group",
                 **kwargs):
        super().__init__()

        self.model = HFVQModel(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            num_vq_embeddings=num_vq_embeddings,
            norm_num_groups=norm_num_groups,
            vq_embed_dim=vq_embed_dim,
            scaling_factor=scaling_factor,
            norm_type=norm_type
        )

    def forward(self, images, **kwargs):
        print(images.shape)
        output = self.model(images.cuda(), return_dict=False)
        print(output.shape)
        troll
        return {"loss": output}

    def __repr__(self):
        s = "VQModel\n"
        s += str(self.model) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
