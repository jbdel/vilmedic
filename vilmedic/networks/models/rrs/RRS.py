import torch.nn as nn
import torch
from collections import OrderedDict
# v4.3.2
from vilmedic.networks.blocks.huggingface.encoder_decoder.evaluation import evaluation
from vilmedic.networks.blocks.huggingface.encoder_decoder.encoder_decoder_model import EncoderDecoderModel


def create_state_dict(ckpt, diff, replace):
    weights = torch.load(ckpt)["model"]
    new_dict = set()
    for k, v in weights.items():
        if diff in k:  # avoid cnn
            new_dict.add((k.replace(diff, replace), v))
    new_dict = OrderedDict(new_dict)
    return new_dict


class RRS(nn.Module):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.enc_dec = EncoderDecoderModel(encoder, decoder)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           )
        return out

    def __repr__(self):
        return "SumHugMono\n" + str(self.enc_dec)
