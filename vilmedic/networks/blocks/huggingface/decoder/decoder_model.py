import torch.nn as nn
import copy
import functools

# v4.3.2
from transformers.models.auto import AutoModelForCausalLM, AutoConfig
from transformers.models.bert_generation import BertGenerationConfig, BertGenerationDecoder
from vilmedic.networks.blocks.huggingface.decoder.beam_search import prepare_inputs_for_generation
from transformers.models.roberta import modeling_roberta

class DecoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, decoder, **kwargs):
        super().__init__()
        if decoder.proto is not None:
            path = decoder.pop('proto')
            dec_config = AutoConfig.from_pretrained(path)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            self.decoder = AutoModelForCausalLM.from_pretrained(path, config=dec_config)
        else:
            dec_config = BertGenerationConfig(**decoder)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            self.decoder = BertGenerationDecoder(dec_config)

        # Evaluation
        self.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation, self.decoder)

        # Inference
        self.generate = self.decoder.generate
        self.config = self.decoder.config

    def forward(self, input_ids, attention_mask, encoder_outputs=None, encoder_attention_mask=None, **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        out = self.decoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           encoder_hidden_states=encoder_outputs,
                           encoder_attention_mask=encoder_attention_mask,
                           labels=input_ids,
                           **kwargs)
        out = vars(out)
        return out

    def __repr__(self):
        s = str(type(self.decoder).__name__) + '(' + str(self.decoder.config) + ')\n'
        return s
