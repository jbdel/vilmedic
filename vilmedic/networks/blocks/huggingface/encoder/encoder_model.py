import torch.nn as nn

# v4.3.2
from transformers.models.auto import AutoModel, AutoConfig
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig


class EncoderModel(nn.Module):
    """
    If proto is mentioned in encoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder model from encoder dict.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__()
        if 'proto' in encoder:
            path = encoder.pop('proto')
            enc_config = AutoConfig.from_pretrained(path)
            self.encoder = AutoModel.from_pretrained(path, config=enc_config)
        else:
            enc_config = BertGenerationConfig(**encoder)
            self.encoder = BertGenerationEncoder(enc_config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           **kwargs)
        out = vars(out)
        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + '(' + str(self.encoder.config) + ')\n'
        return s
