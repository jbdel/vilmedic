import torch.nn as nn
from vilmedic.models.utils import get_n_params
import functools

from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation as evaluation_

from .RRG import RRG
import numpy as np
import copy
import torch
from vilmedic.blocks.rl.SCST import SCST


def evaluation(models, config, dl, **kwargs):
    models = [m.model for m in models]  # Get trained RRG instance
    return evaluation_(models, config, dl, **kwargs)


class RRG_SCST(nn.Module):

    def __init__(self, decoder, cnn, ckpt, dl, scores="ROUGEL", scores_args=None, scores_weights=None, top_k=None,
                 **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)

        # SCST
        self.scst = SCST(decoder=self.model.dec.decoder,
                         dl=dl,
                         scores=scores,
                         scores_args=scores_args,
                         scores_weights=scores_weights,
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
        loss, delta_reward, reward_sampling, sampling_hyp_list = self.scst.forward_sampling(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            reward_greedy=reward_greedy
        )

        # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
        return {"loss": loss,
                "custom_print": "reward_sampling {}, "
                                "delta_reward: {},"
                                "sample_hyp[0]: {},".format(
                    np.mean(reward_sampling),
                    torch.mean(delta_reward),
                    sampling_hyp_list[0]),
                }

    def __repr__(self):
        s = "RRG_PPO\n"
        s += str(self.scst) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
