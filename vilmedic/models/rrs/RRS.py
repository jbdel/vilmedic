import torch.nn as nn
import torch
from collections import OrderedDict
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.huggingface.encoder_decoder.evaluation import evaluation
from vilmedic.blocks.huggingface.encoder_decoder.encoder_decoder_model import EncoderDecoderModel
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel as HFEncoderDecoderModel
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers import RobertaConfig, RobertaModel, RobertaForCausalLM, BertLMHeadModel, BertModel, BertConfig


class RRS(nn.Module):

    def __init__(self, encoder, decoder, dl, **kwargs):
        super().__init__()

        enc_config = BertGenerationConfig(**encoder,
                                          is_decoder=False,
                                          add_cross_attention=False)
        self.enc = BertGenerationEncoder(enc_config)
        self.dec = DecoderModel(decoder)

        # Evaluation
        self.eval_func = evaluation

        # Tokenizer
        self.tokenizer = dl.dataset.tgt_tokenizer

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, epoch=None, iteration=None,
                **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        encoder_outputs = self.enc(input_ids, attention_mask, return_dict=True)
        out = self.dec(input_ids=decoder_input_ids,
                       attention_mask=decoder_attention_mask,
                       encoder_outputs=encoder_outputs.last_hidden_state,
                       encoder_attention_mask=attention_mask,
                       **kwargs)
        return out

    def __repr__(self):
        s = "model: RRS\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
