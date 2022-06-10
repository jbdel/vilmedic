import torch.nn as nn
import copy
import functools

from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel as HFEncoderDecoderModel
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers import RobertaConfig, RobertaModel, RobertaForCausalLM, BertLMHeadModel, BertModel, BertConfig
from vilmedic.models.utils import get_n_params
from vilmedic.blocks.huggingface.decoder.beam_search import beam_search
from vilmedic.blocks.huggingface.decoder.beam_search import prepare_inputs_for_generation


class EncoderDecoderModel(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        if encoder.proto is not None and decoder.proto is not None:
            self.enc_dec = HFEncoderDecoderModel.from_encoder_decoder_pretrained(encoder.pop('proto'),
                                                                                 decoder.pop('proto'))
        else:
            # Encoder
            enc_config = BertConfig(**encoder,
                                    is_decoder=False,
                                    add_cross_attention=False)
            enc = BertGenerationEncoder(enc_config)

            # Decoder
            dec_config = BertConfig(**decoder)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            # dec = BertGenerationDecoder(dec_config)
            dec = BertLMHeadModel(dec_config)

            # Encdec
            config = EncoderDecoderConfig.from_encoder_decoder_configs(enc.config, dec.config, **kwargs)
            self.enc_dec = HFEncoderDecoderModel(config=config)

        self.enc_dec.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation,
                                                                               self.enc_dec.decoder)

        assert self.enc_dec.config.is_encoder_decoder == True

        # Inference
        self.config = self.enc_dec.config

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
        s = str(type(self.enc_dec.encoder).__name__) + '(' + str(self.enc_dec.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
