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
import torch.nn.functional as F
import inspect
import random
from vilmedic.scorers.NLG import ROUGEScorer


def evaluation(models, opts, dl):
    models = [m.model for m in models]  # Get trained RRG instance
    return evaluation_(models, opts, dl)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def scst_loss(input, seq, reward, pad_token_id):
    input = to_contiguous(input).view(-1)
    # HuggingFae TopKLogitsWarper (if top_k > 0) puts -float('inf') for non top_k
    # Padding can be -float('inf') though we have to pad because hyp generation is finished
    # -float('inf') * 0 in mask results in NaN so that doesnt mitigate the issue. need to do:
    input[input == -float("Inf")] = 0.
    #####

    reward = to_contiguous(reward).view(-1)
    mask = (seq > pad_token_id).float()
    mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
    output = - input * reward * mask
    output = torch.sum(output) / torch.sum(mask)
    return output


class RRG_SCST(nn.Module):

    def __init__(self, decoder, cnn, ckpt, dl, logger, score="chexbert", **kwargs):
        super().__init__()

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)
        if 'top_k' in kwargs:
            self.model.dec.decoder.config.top_k = kwargs['top_k']

        self.score = score
        # Scoring
        if score == "chexbert":
            self.chexbert = CheXbert()

        # Tokens
        self.bos_token_id = self.model.dec.decoder.config.bos_token_id
        self.eos_token_id = self.model.dec.decoder.config.eos_token_id
        self.pad_token_id = self.model.dec.decoder.config.pad_token_id

        # Tokenizer
        self.dl = dl
        self.tokenizer = dl.dataset.tgt_tokenizer

        self.eval_func = evaluation
        self.logger = logger

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 1 Greedy
        with torch.no_grad():
            self.model.eval()
            encoder_hidden_states = self.model.encoder(images.cuda())
            out = self.model.dec.decoder.generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
                max_length=seq_len,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                encoder_hidden_states=encoder_hidden_states.data,
            )
            greedy_input_ids = out.sequences
            score_greedy = self.get_reward(greedy_input_ids.data, input_ids)

        # 2. Sampling
        self.model.train()
        encoder_hidden_states = self.model.encoder(images.cuda())
        out = inspect.unwrap(self.model.dec.decoder.generate)(  # inspect.unwrap removes the torch.no_grad() decorator
            self=self.model.dec.decoder,
            input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
            max_length=seq_len,
            num_beams=1,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            encoder_hidden_states=encoder_hidden_states,
            bad_words_ids=[[self.pad_token_id], [self.bos_token_id]]
        )
        samples_ids = out.sequences[:, 1:]
        logits = torch.stack(out.scores, dim=1)
        logits = F.log_softmax(logits, dim=-1)
        sampled_logits = logits.gather(2, samples_ids.unsqueeze(-1))

        # 3. Reward and loss
        score_sampling = self.get_reward(samples_ids.data, input_ids)
        reward = (score_sampling - score_greedy)
        loss = scst_loss(sampled_logits, samples_ids.data, reward, self.pad_token_id)

        if random.randint(0, 10) == 0:
            self.logger.info(greedy_input_ids[0])
            self.logger.info(samples_ids[0])
            self.logger.info(score_greedy)
            self.logger.info(score_sampling)
            self.logger.info(loss)
            self.logger.info("#######")

        return {"loss": loss}

    def get_reward(self, rollout_input_ids, input_ids):

        hyp_list = []
        ref_list = []
        for h, r in zip(rollout_input_ids, input_ids):
            hyp_list.append(self.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ref_list.append(self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        if self.score == "chexbert":
            assert not self.chexbert.training
            chexbert, _ = self.chexbert(hyp_list, ref_list)
            reward = torch.tensor(chexbert["micro avg"]["f1-score"]).cuda()
        else:
            reward = round(ROUGEScorer(rouges=['rouge2']).compute(ref_list, hyp_list), 4)
        return reward

    def __repr__(self):
        s = "RRG_PPO\n"
        # s += str(self.enc) + '\n'
        # s += str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
