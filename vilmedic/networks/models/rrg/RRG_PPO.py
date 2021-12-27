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


def evaluation(models, config, dl, **kwargs):
    models = [m.model for m in models]  # Get trained RRG instance
    return evaluation_(models, config, dl)


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


class RRG_PPO(nn.Module):

    def __init__(self, decoder, cnn, ckpt, ppo=None, **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)
        self.model_ref = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model_ref.load_state_dict(state_dict, strict=True)

        self.v_head = ValueHead()
        del state_dict

        # PPO
        if ppo is None:
            ppo = {}
        self.ppo = PPOTrainer(self.model.encode, self.model.dec, self.model_ref.dec, self.v_head, **ppo)

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
        tokenizer = dataset.tokenizer

        with torch.no_grad():
            # 0. get encoder states
            encoder_hidden_states = self.model.encode(images.cuda())

            # 1. Rollout
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            rollout_input_ids = torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id
            for i in range(seq_len):
                outputs = self.model.dec(
                    rollout_input_ids,
                    attention_mask=rollout_input_ids.new_ones(rollout_input_ids.shape, dtype=torch.long),
                    encoder_outputs=encoder_hidden_states
                )

                next_token_logits = outputs["logits"][:, -1, :]
                next_token_logits = top_k_top_p_filtering(next_token_logits,
                                                          top_k=self.ppo.ppo_params["top_k"],
                                                          top_p=self.ppo.ppo_params["top_p"])
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                rollout_input_ids = torch.cat([rollout_input_ids, next_token.unsqueeze(-1)], dim=-1)

            # 2. Get reward
            hyp_list = []
            ref_list = []
            for h, r in zip(rollout_input_ids, input_ids):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

            chexbert, _ = self.chexbert(hyp_list, ref_list)
            scores = torch.tensor([chexbert["micro avg"]["f1-score"] for _ in range(len(hyp_list))])
            rollout_input_ids = tokenizer(hyp_list, **dataset.tokenizer_args)

            # 3. Optimization
            logprobs, ref_logprobs, values = self.ppo.batched_forward_pass(rollout_input_ids.input_ids.cuda(),
                                                                           rollout_input_ids.attention_mask.cuda(),
                                                                           encoder_hidden_states)

            rewards, non_score_reward, kl_coef = self.ppo.compute_rewards(scores, logprobs, ref_logprobs)

        all_stats = []
        idxs = list(range(batch_size))
        for _ in range(self.ppo.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(batch_size):
                idx = idxs[i]
                train_stats = self.ppo.train_minibatch(seq_len,
                                                       logprobs[idx:idx + 1],
                                                       values[idx:idx + 1],
                                                       rewards[idx:idx + 1],
                                                       rollout_input_ids.input_ids[idx:idx + 1].cuda(),
                                                       rollout_input_ids.attention_mask[idx:idx + 1].cuda(),
                                                       images[idx:idx + 1].cuda())
                all_stats.append(train_stats)

        # Handle stats
        train_stats = stack_dicts(all_stats)
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        stats = self.ppo.get_stats(scores=scores,
                                   logprobs=logprobs,
                                   ref_logprobs=ref_logprobs,
                                   non_score_reward=non_score_reward,
                                   train_stats=train_stats,
                                   kl_coef=kl_coef)

        self.ppo.kl_ctl.update(stats['objective/kl'][-1], batch_size)

        return "objective/kl: {}, objective/kl_coef: {},  objective/score: {}, ppo/mean_non_score_reward: {}".format(
            np.mean(stats['objective/kl']),
            np.mean(stats['objective/kl_coef']),
            np.mean(stats['objective/score']),
            np.mean(stats['ppo/mean_non_score_reward'])
        )

    def __repr__(self):
        s = "RRG_PPO\n"
        # s += str(self.enc) + '\n'
        # s += str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
