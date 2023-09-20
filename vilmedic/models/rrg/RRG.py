import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params

from vilmedic.blocks.vision import VisualEncoder
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation


class RRG(nn.Module):

    def __init__(self, decoder, cnn, dl=None, **kwargs):
        super().__init__()
        # Decoder
        if dl:
            decoder.vocab_size = dl.dataset.seq.tokenizer.vocab_size
        self.dec = DecoderModel(decoder)

        # Encoder
        self.enc = eval(cnn.pop('proto'))(**cnn)

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
    def encode(self, images, images_mask, **kwargs):
        return self.enc.encode(images, images_mask, **kwargs)

    def __repr__(self):
        s = "model: RRG\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
