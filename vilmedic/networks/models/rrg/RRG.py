import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params

# v4.3.2
from transformers.models.bert_generation import BertGenerationConfig, BertGenerationDecoder
from transformers.models.auto import AutoModelForCausalLM, AutoConfig

# from vilmedic.networks.huggingface.beam import beam_search
from vilmedic.networks.blocks.huggingface.beam_dec import beam_search
from vilmedic.networks.blocks.vision import *


def prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)

    # cut decoder_input_ids if past is used
    if past is not None:
        input_ids = input_ids[:, -1:]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past, **model_kwargs}


class RRG(nn.Module):

    def __init__(self, decoder, cnn, **kwargs):
        super().__init__()

        # Decoder
        if 'proto' in decoder:
            path = decoder.pop('proto')
            dec_config = AutoConfig.from_pretrained(path)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            self.dec = AutoModelForCausalLM.from_pretrained(path, config=dec_config)
            # RobertaForCausalLM.generate
            # RobertaForCausalLM.prepare_inputs_for_generation
        else:
            dec_config = BertGenerationConfig(**decoder)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            self.dec = BertGenerationDecoder(dec_config)
            # self.dec.prepare_inputs_for_generation()

        # Encoder
        visual_embedding_dim = cnn.pop("visual_embedding_dim")
        cnn = eval(cnn.pop('proto'))(**cnn)
        visual_projection = nn.Linear(visual_embedding_dim, self.dec.config.hidden_size)
        self.enc = nn.Sequential(cnn, visual_projection)

        self.bos_token_id = self.dec.config.bos_token_id
        self.eos_token_id = self.dec.config.eos_token_id
        self.pad_token_id = self.dec.config.pad_token_id

        # Evaluation
        self.eval_func = beam_search
        self.dec.prepare_inputs_for_generation = prepare_inputs_for_generation

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):
        images = images.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs = self.encoder(images)

        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_hidden_states=encoder_outputs,
                       encoder_attention_mask=None,
                       labels=input_ids,
                       **kwargs)

        out = vars(out)
        return out

    # Necessary for generation
    def encoder(self, images, **kwargs):
        return self.enc(images)

    def __repr__(self):
        s = str(self.enc) + '\n'
        s += str(type(self.dec).__name__) + '(' + str(self.dec.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
