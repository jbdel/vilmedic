import json
import os
import torch
import functools
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from dalle_pytorch import DALLE as PyDALLE
from .VAE import VAE
from vilmedic.networks.models.utils import get_n_params

import torch.nn.functional as F
from einops import rearrange
from .dalle_forward import forward


def evaluation(models, config, dl, **kwargs):
    losses = []
    model = models[0]

    if config.generate_images is not None and config.generate_images:
        pbar = tqdm(dl, total=len(dl))
        for batch in pbar:
            for input_id in batch["input_ids"]:
                print("Inputed text:", dl.dataset.tokenizer.decode(input_id, skip_special_tokens=True))
                print("Generating {} images in dir {}".format(config.num_images, config.ckpt_dir))
                # text_tokens = repeat(input_id.unsqueeze(0), '() n -> b n', b=config.num_images)
                for i in range(config.num_images):
                    print("doing", i)
                    output = model.dalle.generate_images(input_id.unsqueeze(0).cuda(), filter_thres=0.9)
                    print("done output")
                    save_image(output, os.path.join(config.ckpt_dir, '{}.jpg'.format(i)), normalize=True)
                import sys
                sys.exit()

    else:
        pbar = tqdm(dl, total=len(dl))
        for batch in pbar:
            out = model(**batch)
            out['loss'] = out['loss'].mean()
            losses.append(out['loss'].cpu().data.numpy())
        return {'loss': np.ndarray.mean(np.array(losses))}


class DALLE(nn.Module):
    def __init__(self,
                 vae,
                 dalle,
                 forward_batch_size=256,
                 **kwargs):
        super(DALLE, self).__init__()

        assert 'ckpt' in vae, 'please specify vae checkpoint'
        self.vae = VAE(**vae)

        vae_weights = torch.load(vae.pop('ckpt'))["model"]
        vae_weights = {name.replace('module.', ''): param for name, param in vae_weights.items()}
        self.vae.load_state_dict(vae_weights, strict=True)

        self.dalle_config = dalle
        self.dalle = PyDALLE(
            vae=self.vae.vae,  # automatically infer (1) image sequence length and (2) number of image tokens
            **self.dalle_config
        )

        self.fbs = forward_batch_size
        self.dalle.forward = functools.partial(forward, self.dalle)
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, **kwargs):
        images = images.cuda()
        input_ids = input_ids.cuda()

        bs = images.shape[0]
        forward_logits = []
        forward_images = []
        forward_texts = []

        # Forward pass
        for i in range(int(bs / min(self.fbs, bs))):
            input_id = input_ids[i * self.fbs:(i + 1) * self.fbs]
            image = images[i * self.fbs:(i + 1) * self.fbs]
            logits, text, image = self.dalle(input_id, image, return_loss=False)

            forward_logits.append(logits)
            forward_images.append(image)
            forward_texts.append(text)

        forward_logits = torch.cat(forward_logits)
        forward_images = torch.cat(forward_images)
        forward_texts = torch.cat(forward_texts)

        # Compute loss
        offsetted_image = forward_images + self.dalle.num_text_tokens
        labels = torch.cat((forward_texts[:, 1:], offsetted_image), dim=1)
        logits = rearrange(forward_logits, 'b n c -> b c n')
        loss_text = F.cross_entropy(logits[:, :, :self.dalle.text_seq_len], labels[:, :self.dalle.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.dalle.text_seq_len:], labels[:, self.dalle.text_seq_len:])
        loss = (loss_text + self.dalle.loss_img_weight * loss_img) / (self.dalle.loss_img_weight + 1)

        return {'loss': loss}

    def __repr__(self):
        s = str(type(self.dalle).__name__) + '(' + str(
            json.dumps(dict(self.dalle_config), indent=4, sort_keys=True)) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
