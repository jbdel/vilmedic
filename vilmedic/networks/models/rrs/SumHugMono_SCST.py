import logging
import functools
import torch.nn.functional as F
import inspect
import random
import torch.nn as nn
import torch
from collections import OrderedDict
# v4.3.2
from vilmedic.networks.blocks.huggingface.encoder_decoder.evaluation import evaluation
from vilmedic.networks.blocks.huggingface.encoder_decoder.encoder_decoder_model import EncoderDecoderModel
from vilmedic.scorers.NLG import ROUGEScorer
from vilmedic.scorers.scores import CheXbert
from transformers.models.roberta.modeling_roberta import RobertaForCausalLM
import numpy as np


def create_state_dict(ckpt, diff, replace):
    weights = torch.load(ckpt)["model"]
    new_dict = set()
    for k, v in weights.items():
        if diff in k:  # avoid cnn
            new_dict.add((k.replace(diff, replace), v))
    new_dict = OrderedDict(new_dict)
    return new_dict


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)

    # cut decoder_input_ids if past is used
    if past is not None:
        input_ids = input_ids[:, -1:]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past, **model_kwargs}


def scst_loss(input, seq, reward, pad_token_id):
    # HuggingFace TopKLogitsWarper (if top_k > 0) puts -float('inf') for non top_k (or if we use bad_words_ids)
    # Padding can then have logits -float('inf') though we have to pad because hyp generation is finished
    # -float('inf') * 0 masking results in NaN so that doesnt mitigate the issue. need to do:
    input[input == -float("Inf")] = 0.
    #####

    # reward
    N, L = input.shape[:2]
    reward = reward.view(N, 1, 1).expand_as(input)
    input = -input * reward

    # masking
    mask = (seq > pad_token_id).float()
    output = input.view(-1) * mask.view(-1)

    # mean
    output = torch.sum(output) / torch.sum(mask)
    return output


class SumHugMono_SCST(nn.Module):

    def __init__(self, encoder, decoder, dl, logger=None, score="chexbert", top_k=None, **kwargs):
        super().__init__()
        self.enc_dec = EncoderDecoderModel(encoder, decoder)

        # Do we have pretrained ?
        if 'ckpt' in kwargs and 'enc' in kwargs['ckpt']:
            st = create_state_dict(kwargs['ckpt']['enc'], diff='linguistic.encoder.', replace='')
            self.enc_dec.enc_dec.encoder.load_state_dict(st, strict=True)
            print("loaded")
        elif 'ckpt' in kwargs and 'dec' in kwargs['ckpt']:
            st = create_state_dict(kwargs['ckpt']['dec'], diff='linguistic.encoder.', replace='roberta.')
            self.enc_dec.enc_dec.decoder.load_state_dict(st, strict=False)
        elif 'ckpt' in kwargs:
            self.load_state_dict(torch.load(kwargs["ckpt"])["model"], strict=True)

        self.score = score
        self.top_k = top_k
        # Scoring
        if score == "chexbert":
            self.chexbert = CheXbert()

        # Tokens
        self.bos_token_id = self.enc_dec.enc_dec.decoder.config.bos_token_id
        self.eos_token_id = self.enc_dec.enc_dec.decoder.config.eos_token_id
        self.pad_token_id = self.enc_dec.enc_dec.decoder.config.pad_token_id

        self.enc_dec.enc_dec.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation,
                                                                                       self.enc_dec.enc_dec.decoder)

        # Tokenizer
        self.dl = dl
        self.tokenizer = dl.dataset.tgt_tokenizer
        self.logger = logger or logging.getLogger(__name__)

        # Evaluation
        self.eval_func = evaluation

        self.train_rewards = []

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        decoder_input_ids = decoder_input_ids.cuda()

        # 1 Greedy
        with torch.no_grad():
            self.enc_dec.eval()
            encoder_outputs = self.enc_dec.enc_dec.encoder(input_ids, attention_mask)
            hidden_states = encoder_outputs[0]
            out = self.enc_dec.enc_dec.decoder.generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
                max_length=seq_len,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                forced_eos_token_id=True,
            )
            greedy_input_ids = out.sequences
            score_greedy = self.get_reward(greedy_input_ids.data, decoder_input_ids)

        # 2. Sampling
        # self.enc_dec.train()
        self.enc_dec.eval()
        encoder_outputs = self.enc_dec.enc_dec.encoder(input_ids, attention_mask)
        hidden_states = encoder_outputs[0]
        out = inspect.unwrap(self.enc_dec.enc_dec.decoder.generate)(
            # inspect.unwrap removes the torch.no_grad() decorator
            self=self.enc_dec.enc_dec.decoder,
            do_sample=True,
            top_k=self.top_k,
            input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
            max_length=seq_len,
            num_beams=1,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            bad_words_ids=[[self.pad_token_id], [self.bos_token_id]],
            forced_eos_token_id=True,
        )
        samples_ids = out.sequences[:, 1:]
        logits = torch.stack(out.scores, dim=1)
        logits = F.log_softmax(logits, dim=-1)
        sampled_logits = logits.gather(2, samples_ids.unsqueeze(-1))

        # 3. Reward and loss
        score_sampling = self.get_reward(samples_ids.data, decoder_input_ids)
        reward = torch.tensor(score_sampling).cuda() - torch.tensor(score_greedy).cuda()
        loss = scst_loss(sampled_logits, samples_ids.data, reward, self.pad_token_id)

        self.train_rewards.append(score_greedy)
        self.logger.info(np.mean(np.array(self.train_rewards)[-500:]))
        if random.randint(0, 500) == 0:
            self.logger.info("#" * 10)
            self.logger.info(
                self.tokenizer.decode(greedy_input_ids.data[0], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False))
            self.logger.info(
                self.tokenizer.decode(samples_ids.data[0], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False))
            self.logger.info("#" * 10)
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
            reward = ROUGEScorer(rouges=[self.score]).compute(ref_list, hyp_list)[1]
        return reward

    def __repr__(self):
        return "SumHugMono\n" + str(self.enc_dec)
