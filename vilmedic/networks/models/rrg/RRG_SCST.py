import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
import functools
# v4.3.2

from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.huggingface.decoder.evaluation import evaluation as evaluation_
from vilmedic.scorers.scores import CheXbert
from .RRG import RRG
import numpy as np
import copy
import torch
from transformers import top_k_top_p_filtering
import torch.nn.functional as F
from torch.nn import Identity
from .PPOTrainer import PPOTrainer, stack_dicts
import random


def evaluation(models, opts, dl):
    models = [m.model for m in models]  # Get trained RRG instance
    return evaluation_(models, opts, dl)


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""

    def __init__(self):
        super().__init__()
        self.detach_head = False
        self.summary = nn.Linear(768, 1)
        self.activation = Identity()
        self.first_dropout = nn.Dropout(0.1)
        self.last_dropout = Identity()
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class RRG_SCST(nn.Module):

    def __init__(self, decoder, cnn, ckpt, ppo=None, **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)

        # Scoring
        self.chexbert = CheXbert()

        # Tokenizers
        self.bos_token_id = self.model.dec.decoder.config.bos_token_id
        self.eos_token_id = self.model.dec.decoder.config.eos_token_id

        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):
        # Tokenizer
        assert 'dl' in kwargs
        dataset = kwargs["dl"].dataset
        tokenizer = dataset.tgt_tokenizer

        with torch.no_grad():
            # 0. get encoder states
            self.model.eval()
            encoder_hidden_states = self.model.encoder(images.cuda())

            # 1. Rollout
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            rollout_input_ids = torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id
            rollout_logits = torch.zeros((batch_size, seq_len), dtype=torch.long).cuda()

            for i in range(seq_len):
                outputs = self.model.dec(
                    rollout_input_ids,
                    attention_mask=rollout_input_ids.new_ones(rollout_input_ids.shape, dtype=torch.long),
                    encoder_outputs=encoder_hidden_states
                )

                next_token_logits = outputs["logits"][:, -1, :]

                # Sample
                it = torch.distributions.Categorical(logits=next_token_logits.detach()).sample()
                sample_logits = next_token_logits.gather(1, it.unsqueeze(1))
                rollout_input_ids = torch.cat([rollout_input_ids, it.unsqueeze(1)], dim=-1)
                print(rollout_input_ids)
                troll

            # 2. Get reward
            hyp_list = []
            ref_list = []
            for h, r in zip(rollout_input_ids, input_ids):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

            chexbert, _ = self.chexbert(hyp_list, ref_list)
            scores = torch.tensor([chexbert["micro avg"]["f1-score"] for _ in range(len(hyp_list))])
            rollout_input_ids = tokenizer(hyp_list, **dataset.tokenizer_args)

        return "{}".format(0
                           )

    def __repr__(self):
        s = "RRG_PPO\n"
        # s += str(self.enc) + '\n'
        # s += str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
