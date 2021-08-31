from ..vision.cnn import CNN
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert_generation import BertGenerationConfig


class CNNTransformer(nn.Module):
    def __init__(self, cnn, transformer, **kwargs):
        super().__init__()
        self.cnn = CNN(**cnn)
        transformer = BertGenerationConfig(transformer)
        self.transformer = BertEncoder(transformer)

    def forward(self):
        pass