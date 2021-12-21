import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

# v4.3.2
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from transformers import EncoderDecoderModel
from .beam import beam_search


class Seq2SeqHug(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        if 'proto' in decoder and 'proto' in encoder:
            self.enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder.pop('proto'),
                                                                               decoder.pop('proto'))

            self.enc = self.enc_dec.encoder
            self.dec = self.enc_dec.decoder
        else:
            # Encoder
            encoder = vars(encoder)
            enc_config = BertGenerationConfig(**encoder)
            self.enc = BertGenerationEncoder(enc_config)

            # Decoder
            decoder = vars(decoder)
            dec_config = copy.deepcopy(enc_config)
            dec_config.update(decoder)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            self.dec = BertGenerationDecoder(dec_config)

            # Encdec
            self.enc_dec = EncoderDecoderModel(encoder=self.enc, decoder=self.dec)

        # Tokens
        self.bos_token_id = self.dec.config.bos_token_id
        self.eos_token_id = self.dec.config.eos_token_id
        self.pad_token_id = self.dec.config.pad_token_id

        # Evaluation
        self.eval_func = beam_search
        self.generate_config = {}

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        input_ids = input_ids.cuda()
        decoder_input_ids = decoder_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        decoder_attention_mask = decoder_attention_mask.cuda()
        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=decoder_input_ids)
        out = vars(out)
        return out

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = str(type(self.enc_dec.encoder).__name__) + '(' + str(self.enc_dec.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
