import torch.nn as nn
from vilmedic.models.utils import get_n_params
import functools
import os
from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation as evaluation_
import glob

from .RRG import RRG
import numpy as np
import copy
import torch
from vilmedic.blocks.rl.SCST import SCST


def evaluation(models, config, dl, **kwargs):
    models = [m.model if not isinstance(m, nn.DataParallel) else m.module.model for m in
              models]  # Get trained RRG instance
    return evaluation_(models, config, dl, **kwargs)


def get_ckpt(ckpt):
    if ".pth" in ckpt:
        return ckpt
    elif os.path.isdir(ckpt):
        ckpt = glob.glob(os.path.join(ckpt, "*.pth"))
        assert len(ckpt) == 1
        return ckpt
    else:
        raise FileNotFoundError(ckpt)


class RRG_SCST(nn.Module):

    def __init__(self, decoder, cnn, dl, scores="ROUGEL", ckpt=None, scores_args=None, scores_weights=None, top_k=None,
                 use_nll=False, **kwargs):
        super().__init__()

        # Models

        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), dl=dl, **kwargs)
        if ckpt:
            state_dict = torch.load(get_ckpt(ckpt))["model"]
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=True)

        # SCST
        self.scst = SCST(decoder=self.model.dec.decoder,
                         dl=dl,
                         scores=scores,
                         scores_args=scores_args,
                         scores_weights=scores_weights,
                         use_nll=use_nll,
                         top_k=top_k)

        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, images_mask=None, encoder_outputs=None, **kwargs):
        # 1 Greedy
        with torch.no_grad():
            self.model.eval()
            encoder_hidden_states, encoder_attention_mask = self.model.encode(images.cuda(), images_mask, **kwargs)
            reward_greedy, greedy_hyp_list, ref_list = self.scst.forward_greedy(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )

        # 2. Sampling
        self.model.train()
        encoder_hidden_states, encoder_attention_mask = self.model.encode(images.cuda(), images_mask, **kwargs)
        loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list = self.scst.forward_sampling(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            reward_greedy=reward_greedy
        )

        return {"loss": loss,
                "custom_print": "reward_sampling {}, "
                                "delta_reward: {}".format(torch.mean(torch.tensor(reward_sampling)),
                                                          torch.mean(torch.tensor(delta_reward))),
                }

    def __repr__(self):
        s = "RRG_SCST\n"
        s += str(self.scst) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
