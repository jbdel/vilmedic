import torch
import torch.nn as nn
from vilmedic.models.utils import get_n_params
import numpy as np

from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.decoder_model import DecoderModel
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation
from vilmedic.blocks.vision.eeg_modeling.dense_inception import DenseInception

from einops import rearrange


class RRG_EEG(nn.Module):

    def __init__(self, decoder, dl, pretrained=False, freeze=False,
                 **kwargs):
        super().__init__()
        # Decoder
        self.dec = DecoderModel(decoder)

        # Encoder
        eeg_enc = DenseInception(data_shape=(2400, 19))
        if pretrained:
            pretrained_pth = "/home/ksaab/Documents/eeg_fully_supervised/results/lpch_mini2/scale_1/cv_3/best.pth.tar"
            state_dict = torch.load(pretrained_pth)["state_dict"]
            og_state_dict = eeg_enc.state_dict()
            # clean up state dict keys
            state_dict_ = {}
            for key in state_dict:
                if "fc2" not in key:
                    state_dict_[key.split("dense_inception.")[-1]] = state_dict[key]
                else:
                    state_dict_[key.split("dense_inception.")[-1]] = og_state_dict[
                        key.split("dense_inception.")[-1]
                    ]
            eeg_enc.load_state_dict(state_dict_)
        eeg_enc.fc2 = torch.nn.Identity()
        if freeze:
            for param in eeg_enc.parameters():
                param.requires_grad = False

        visual_projection = nn.Linear(360, self.dec.decoder.config.hidden_size)
        self.enc = nn.Sequential(eeg_enc, visual_projection)

        # Evaluation
        self.eval_func = evaluation
        self.tokenizer = dl.dataset.tokenizer

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None,
                encoder_attention_mask=None, epoch=None, iteration=None, **kwargs):
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(images=images, **kwargs)

        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_outputs=encoder_outputs,
                       encoder_attention_mask=encoder_attention_mask,
                       **kwargs)
        # print(self.tokenizer.decode(torch.argmax(out["logits"][0], dim=-1), skip_special_tokens=True,
        #                             clean_up_tokenization_spaces=False))

        return out

    # Necessary for generation
    def encode(self, images, **kwargs):
        eeg = images
        if torch.cuda.is_available():
            eeg = eeg.cuda()
        feature = self.enc(eeg)
        mask = (torch.sum(torch.abs(feature), dim=-1) != 0)
        return feature, mask

    def __repr__(self):
        s = "model: RRG_EEG\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
