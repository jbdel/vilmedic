import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation
from einops import rearrange


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

    def forward(self, input_ids, attention_mask, images, images_mask=None, encoder_outputs=None,
                encoder_attention_mask=None, **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(images, images_mask, **kwargs)

        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_outputs=encoder_outputs,
                       encoder_attention_mask=encoder_attention_mask,
                       **kwargs)

        return out

    # Necessary for generation
    def encode(self, images, images_mask=None, **kwargs):
        images = images.cuda()
        # Single-image forward pass
        if len(images.shape) == 4:
            feature = self.enc(images)
            feature_mask = (torch.sum(torch.abs(feature), dim=-1) != 0)
            return feature, feature_mask

        assert len(images.shape) == 5, "wrong images shape"

        # Multi-image forward pass
        images = rearrange(images, 'd0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4')
        feature = self.enc(images)

        # Masking features of empty images
        num_images = images.shape[1]
        feature = feature.view(int(feature.shape[0] / num_images), num_images, feature.shape[-2], feature.shape[-1])
        feature = feature * images_mask.unsqueeze(-1).unsqueeze(-1).cuda()

        # Creating feature-wise attention mask
        feature = rearrange(feature, 'd0 d1 d2 d3 -> d0 (d1 d2) d3')
        feature_mask = (torch.sum(torch.abs(feature), dim=-1) != 0)
        return feature, feature_mask

    def __repr__(self):
        s = "model: RRG\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
