import torch.nn as nn

# v4.3.2
from transformers.models.auto import AutoModel, AutoConfig
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig

from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertPooler


class EncoderModel(nn.Module):
    """
    If proto is mentioned in encoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder model from encoder dict.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__()
        if encoder.proto is not None:
            path = encoder.pop('proto')
            enc_config = AutoConfig.from_pretrained(path)
            self.encoder = AutoModel.from_pretrained(path, config=enc_config)
        else:
            enc_config = BertGenerationConfig(**encoder)
            self.encoder = BertGenerationEncoder(enc_config)

        if encoder.add_pooling_layer:  # 4 info: roberta already has a pooler layer
            self.pooler = BertPooler(encoder)

    def forward(self, input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs):

        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           past_key_values=past_key_values,
                           use_cache=use_cache,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=True,
                           )

        if hasattr(self, "pooler"):
            pooled_output = self.pooler(hidden_states=out.last_hidden_state)
            setattr(out, "pooler_output", pooled_output)

        out = vars(out)
        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + '(' + str(self.encoder.config) + ')\n'
        return s
