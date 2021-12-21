import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

from .VQABert.VisualBert import VisualBERTBase
from .VQABert.beam import beam_search
from ..vision.cnn import CNN
import torch

class VQA2SeqHug(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, cnn, encoder, **kwargs):
        super().__init__()

        # Encoder
        self.enc = VisualBERTBase(**encoder)
        self.backbone = CNN(**cnn).eval()
        # self.visual_projection = nn.Linear(encoder.pop("visual_embedding_dim"), self.enc.config.hidden_size)
        self.eval_func = beam_search
        self.generate_config = {}

    def forward(self, input_ids, images, attention_mask, labels, **kwargs):
        features, _ = self.backbone(images.cuda())
        # features = features.detach()
        out = self.enc(input_ids=input_ids.cuda(),
                       input_mask=attention_mask.cuda(),
                       attention_mask=attention_mask.cuda(),
                       visual_embeddings=features,
                       labels=labels.cuda(),
                       )
        return out
    #
    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.backbone.network.freeze:
            self.backbone.train(False)
        return self

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = str(type(self.enc).__name__) + '(' + str(self.enc.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
