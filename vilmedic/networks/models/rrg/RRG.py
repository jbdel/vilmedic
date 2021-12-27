import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
# v4.3.2

from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.networks.blocks.huggingface.decoder.evaluation import evaluation


class RRG(nn.Module):

    def __init__(self, decoder, cnn, **kwargs):
        super().__init__()

        # Decoder
        self.dec = DecoderModel(decoder)

        # Encoder
        visual_embedding_dim = cnn.pop("visual_embedding_dim")
        cnn = eval(cnn.pop('proto'))(**cnn)
        visual_projection = nn.Linear(visual_embedding_dim, self.dec.decoder.config.hidden_size)
        self.enc = nn.Sequential(cnn, visual_projection)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):
        images = images.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs = self.encode(images)

        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_outputs=encoder_outputs,
                       **kwargs)

        return out

    # Necessary for generation
    def encode(self, images, **kwargs):
        images = images.cuda()
        return self.enc(images)

    def __repr__(self):
        s = "RRG\n"
        s += str(self.enc) + '\n'
        s += str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
