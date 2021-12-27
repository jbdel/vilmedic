import inspect

import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
# v4.3.2

from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.huggingface.decoder.evaluation import evaluation as evaluation_
from vilmedic.scorers.scores import CheXbert
from .RRG import RRG
import numpy as np
import copy
import torch
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


class RRG_PPO_x(nn.Module):

    def __init__(self, decoder, cnn, ckpt, dl, ppo=None, **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)
        self.model_ref = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model_ref.load_state_dict(state_dict, strict=True)
        # self.model_ref = self.model
        self.v_head = ValueHead()
        del state_dict

        # PPO
        if ppo is None:
            ppo = {}
        self.ppo = PPOTrainer(self.model.encode, self.model.dec, self.model_ref.dec, self.v_head, **ppo)

        # Scoring
        self.chexbert = CheXbert()

        # Tokens
        self.bos_token_id = self.model.dec.decoder.config.bos_token_id
        self.eos_token_id = self.model.dec.decoder.config.eos_token_id
        self.pad_token_id = self.model.dec.decoder.config.pad_token_id

        # Tokenizer
        self.dl = dl
        self.tokenizer = dl.dataset.tokenizer

        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):

        # 1 Sampling
        with torch.no_grad():
            self.model.eval()
            encoder_hidden_states = self.model.encode(images.cuda())
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            out = self.model.dec.decoder.generate(
                    input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
                    max_length=seq_len,
                    num_beams=1,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    do_sample=True,
                    encoder_hidden_states=encoder_hidden_states,
                    top_k=self.ppo.ppo_params["top_k"],
                    top_p=self.ppo.ppo_params["top_p"],
            )
            samples_ids = out.sequences
            samples_ids_attention_mask = (samples_ids != self.pad_token_id).float()
            samples_ids_seq_len = samples_ids.shape[1]

            # 2. Get reward
            score_greedy = self.get_reward(samples_ids.data, input_ids)

            # 3. Optimization
            logprobs, ref_logprobs, values = self.ppo.batched_forward_pass(samples_ids,
                                                                           samples_ids_attention_mask,
                                                                           encoder_hidden_states)

            rewards, non_score_reward, kl_coef = self.ppo.compute_rewards(score_greedy, logprobs, ref_logprobs)

        all_stats = []
        idxs = list(range(batch_size))
        for _ in range(self.ppo.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(batch_size):
                idx = idxs[i]
                train_stats = self.ppo.train_minibatch(samples_ids_seq_len,
                                                       logprobs[idx:idx + 1],
                                                       values[idx:idx + 1],
                                                       rewards[idx:idx + 1],
                                                       samples_ids[idx:idx + 1],
                                                       samples_ids_attention_mask[idx:idx + 1].cuda(),
                                                       images[idx:idx + 1].cuda())
                all_stats.append(train_stats)

        # Handle stats
        train_stats = stack_dicts(all_stats)
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        stats = self.ppo.get_stats(scores=score_greedy,
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

    def get_reward(self, rollout_input_ids, input_ids):
        assert not self.chexbert.training

        hyp_list = []
        ref_list = []
        for h, r in zip(rollout_input_ids, input_ids):
            hyp_list.append(self.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ref_list.append(self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        chexbert, _ = self.chexbert(hyp_list, ref_list)
        reward = torch.tensor(chexbert["micro avg"]["f1-score"]).cuda()
        return reward

    def __repr__(self):
        s = "RRG_PPO\n"
        # s += str(self.enc) + '\n'
        # s += str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
