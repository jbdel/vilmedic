import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from dalle_pytorch import DALLE as PyDALLE
from vilmedic.networks.models.dalle.VAE import VAE
from vilmedic.networks.models.utils import get_n_params


def evaluation(models, opts, dl):
    losses = []
    model = models[0]

    if opts.generate_images is not None and opts.generate_images:
        pbar = tqdm(dl, total=len(dl))
        for batch in pbar:
            for input_id in batch["input_ids"]:
                # text_tokens = repeat(input_id.unsqueeze(0), '() n -> b n', b=opts.num_images)
                for i in range(opts.num_images):
                    print(dl.dataset.tokenizer.decode(input_id, skip_special_tokens=True))
                    output = model.dalle.generate_images(input_id.unsqueeze(0).cuda(), filter_thres=0.9)
                    save_image(output, os.path.join(opts.ckpt_dir, '1.jpg'), normalize=True)
                    if i == 5:
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
                 **kwargs):
        super(DALLE, self).__init__()

        assert 'ckpt' in vae, 'please specify vae checkpoint'
        self.vae = VAE(**vae)
        self.vae.load_state_dict(torch.load(vae.pop('ckpt')), strict=True)

        self.dalle_config = dalle
        self.dalle = PyDALLE(
            vae=self.vae.vae,  # automatically infer (1) image sequence length and (2) number of image tokens
            **self.dalle_config
        )

        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, **kwargs):
        loss = self.dalle(input_ids.cuda(), images.cuda(), mask=attention_mask.cuda(), return_loss=True)
        return {'loss': loss}

    def __repr__(self):
        s = str(type(self.dalle).__name__) + '(' + str(
            json.dumps(dict(self.dalle_config), indent=4, sort_keys=True)) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
