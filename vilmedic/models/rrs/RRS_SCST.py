import copy
import torch
import torch.nn as nn
from .RRS import RRS

from vilmedic.models.utils import get_n_params
from vilmedic.blocks.rl.SCST import SCST
from vilmedic.blocks.huggingface.encoder_decoder.evaluation import evaluation as evaluation_


def evaluation(models, config, dl, **kwargs):
    models = [m.model if not isinstance(m, nn.DataParallel) else m.module.model for m in
              models]  # Get trained RRS instance
    return evaluation_(models, config, dl, **kwargs)


class RRS_SCST(nn.Module):

    def __init__(self, encoder, decoder, ckpt, dl, scores="ROUGEL", scores_args=None, scores_weights=None, top_k=None,
                 use_nll=False, **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRS(copy.deepcopy(encoder), copy.deepcopy(decoder), dl, **kwargs)
        self.model.load_state_dict(state_dict, strict=True)

        self.encoder = self.model.enc
        self.decoder = self.model.dec.decoder

        # SCST
        self.scst = SCST(decoder=self.decoder,
                         dl=dl,
                         scores=scores,
                         scores_args=scores_args,
                         scores_weights=scores_weights,
                         use_nll=use_nll,
                         top_k=top_k)

        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        # 1 Greedy
        with torch.no_grad():
            self.model.eval()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            encoder_outputs = self.encoder(input_ids, attention_mask, return_dict=True)

            reward_greedy, greedy_hyp_list, ref_list = self.scst.forward_greedy(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask
            )

        # 2. Sampling
        self.model.train()
        encoder_outputs = self.encoder(input_ids, attention_mask, return_dict=True)
        loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list = self.scst.forward_sampling(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            reward_greedy=reward_greedy
        )

        return {"loss": loss,
                "custom_print": "reward_sampling {}, "
                                "delta_reward: {},".format(torch.mean(torch.tensor(reward_sampling)),
                                                           torch.mean(torch.tensor(delta_reward))
                                                           ),
                }

    def __repr__(self):
        s = "RRS_SCST\n"
        s += str(self.scst) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
