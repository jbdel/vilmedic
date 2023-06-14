import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation
from einops import rearrange


class RRG_2D_CT(nn.Module):

    def __init__(self, decoder, cnn, dl=None, **kwargs):
        super().__init__()
        # Decoder
        if dl:
            decoder.vocab_size = dl.dataset.seq.tokenizer.vocab_size
        self.dec = DecoderModel(decoder)

        # Encoder
        visual_embedding_dim = cnn.pop("visual_embedding_dim", None)
        cnn = eval(cnn.pop('proto'))(**cnn)
        if visual_embedding_dim:
            visual_projection = nn.Linear(visual_embedding_dim, self.dec.decoder.config.hidden_size)
        else:
            visual_projection = nn.Identity()

        self.enc = nn.Sequential(cnn, visual_projection)

        # Evaluation
        self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, images_mask=None, encoder_outputs=None,
                encoder_attention_mask=None, epoch=None, iteration=None, **kwargs):

        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
            images = images.cuda()

        outputs = []
        for i in range(images.size(-1)):
            slice_ = images[..., i]
            output = self.enc(slice_)
            outputs.append(output)

        features = torch.stack(outputs, dim=1)
        feature_mask = (torch.sum(torch.abs(features), dim=-1) != 0)
        return features, feature_mask

    def __repr__(self):
        s = "model: RRG_2D_CT\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
