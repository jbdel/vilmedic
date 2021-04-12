import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

# v4.3.2
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from transformers import EncoderDecoderModel, EncoderDecoderConfig

from .CNNTransformer import CNNTransformer
from .beam import beam_search


class Im2SeqHug(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()

        # Encoder
        self.enc = CNNTransformer(**encoder)

        # Decoder
        decoder = vars(decoder)
        dec_config = copy.deepcopy(enc_config)
        dec_config.update(decoder)
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        self.dec = BertGenerationDecoder(dec_config)

        # Encdec
        self.enc_dec = EncoderDecoderModel(encoder=self.enc, decoder=self.dec)
        self.bos_token_id = enc_config.bos_token_id
        self.eos_token_id = enc_config.eos_token_id
        self.pad_token_id = enc_config.pad_token_id

        # Evaluation
        self.eval_func = beam_search

        # Embeddings
        if 'src_emb' in kwargs:
            set_embeddings(kwargs['src_emb'], self.enc.embeddings.word_embeddings)
        if 'tgt_emb' in kwargs:
            set_embeddings(kwargs['tgt_emb'], self.dec.bert.embeddings.word_embeddings)

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

    def get_cls(self, encoder_proto, decoder_proto):
        ret = []
        for proto in [encoder_proto, decoder_proto]:
            if 'BertGeneration' in proto:
                ret.extend([BertGenerationConfig])
                if 'Encoder' in proto:
                    ret.extend([BertGenerationEncoder])
                elif 'Decoder' in proto:
                    ret.extend([BertGenerationDecoder])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        assert len(ret) == 4
        return ret
