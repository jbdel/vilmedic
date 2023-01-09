import torch
import copy
import torch.nn as nn
from vilmedic.models.utils import get_n_params
from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.encoder.encoder_model import EncoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation
from vilmedic.models.rrg.RRG import RRG
from einops import rearrange


class RRG_MULTI(nn.Module):

    def __init__(self, encoder, decoder, cnn, dl, **kwargs):
        super().__init__()

        # Classical RRG
        decoder.vocab_size = dl.dataset.seq2seq.tgt.tokenizer.vocab_size
        self.mono_rrg = RRG(decoder, cnn, dl=None, **kwargs)

        # Linguistic encoder
        encoder.vocab_size = dl.dataset.seq2seq.src.tokenizer.vocab_size
        self.ling_enc = EncoderModel(encoder)

        # Evaluation
        self.eval_func = evaluation
        self.dec = self.mono_rrg.dec

    def forward(self,
                input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                images, images_mask=None,
                encoder_outputs=None, encoder_attention_mask=None,
                epoch=None, iteration=None, **kwargs):

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            decoder_input_ids = decoder_input_ids.cuda()
            decoder_attention_mask = decoder_attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(input_ids,
                                                                  attention_mask,
                                                                  images,
                                                                  images_mask,
                                                                  **kwargs)

        # print(decoder_input_ids.shape)
        # print(decoder_attention_mask.shape)
        # print(encoder_outputs.shape)
        # print(encoder_attention_mask.shape)
        # troll
        out = self.mono_rrg.dec(input_ids=decoder_input_ids,
                                attention_mask=decoder_attention_mask,
                                encoder_outputs=encoder_outputs,
                                encoder_attention_mask=encoder_attention_mask,
                                **kwargs)

        return out

    # Necessary for generation
    def encode(self, input_ids, attention_mask, images, images_mask=None, **kwargs):
        # Visual features
        visual_features, visual_mask = self.mono_rrg.encode(images, images_mask=images_mask)

        # Linguistic features
        encoder_outputs = self.ling_enc(input_ids, attention_mask, return_dict=True)

        # Concatenation of features and masks
        encoder_hidden_states = torch.cat((encoder_outputs.last_hidden_state, visual_features), dim=1)
        encoder_attention_mask = torch.cat((attention_mask, visual_mask), dim=-1)

        return encoder_hidden_states, encoder_attention_mask

    def __repr__(self):
        s = "model: RRG\n"
        s += "(cnn):" + str(self.mono_rrg.enc) + '\n'
        s += "(enc):" + str(self.ling_enc) + '\n'
        s += "(dec):" + str(self.mono_rrg.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
