import torch.nn as nn
import torch
from collections import OrderedDict
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.encoder.encoder_model import EncoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation


class RRS(nn.Module):

    def __init__(self, encoder, decoder, dl, **kwargs):
        super().__init__()

        # Encoder
        encoder.vocab_size = dl.dataset.src.tokenizer.vocab_size
        self.enc = EncoderModel(encoder)

        # Decoder
        decoder.vocab_size = dl.dataset.tgt.tokenizer.vocab_size
        self.dec = DecoderModel(decoder)

        # Evaluation
        self.eval_func = evaluation

        # Tokenizer
        self.tokenizer = dl.dataset.tgt_tokenizer

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, encoder_outputs=None,
                encoder_attention_mask=None,
                epoch=None, iteration=None,
                **kwargs):

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(input_ids, attention_mask, **kwargs)

        out = self.dec(input_ids=decoder_input_ids,
                       attention_mask=decoder_attention_mask,
                       encoder_outputs=encoder_outputs,
                       encoder_attention_mask=encoder_attention_mask,
                       **kwargs)

        return out

    def encode(self, input_ids, attention_mask, **kwargs):
        encoder_outputs = self.enc(input_ids, attention_mask, return_dict=True)
        return encoder_outputs.last_hidden_state, attention_mask

    def __repr__(self):
        s = "model: RRS\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
