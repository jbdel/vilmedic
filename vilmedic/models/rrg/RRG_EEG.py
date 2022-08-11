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
        eeg_enc = DenseInception(num_classes=768, data_shape=(12000, 19))
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
        # eeg_enc.fc2 = torch.nn.Identity()
        if freeze:
            for param in eeg_enc.parameters():
                param.requires_grad = False

        # visual_projection = nn.Linear(360, self.dec.decoder.config.hidden_size)
        # self.enc = nn.Sequential(eeg_enc, visual_projection)
        self.enc = eeg_enc

        # Evaluation
        self.eval_func = evaluation
        self.tokenizer = dl.dataset.tokenizer

    def forward(self, input_ids, attention_mask, images, images_mask=None, encoder_outputs=None,
                encoder_attention_mask=None, epoch=None, iteration=None, **kwargs):

        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(images, images_mask, **kwargs)

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_outputs=encoder_outputs,
                       encoder_attention_mask=encoder_attention_mask,
                       **kwargs)
        # print(self.tokenizer.decode(torch.argmax(out["logits"][0], dim=-1), skip_special_tokens=True,
        #                             clean_up_tokenization_spaces=False))

        return out

    # Necessary for generation
    def encode(self, images, images_mask=None, **kwargs):
        eeg = images.cuda()
        images_mask = images_mask.cuda()

        # eeg.shape torch.Size([bs, 30, 60, 19, 200])
        eeg = rearrange(eeg, 'd0 d1 d2 d3 d4 -> (d0 d1) (d2 d4) d3')
        eeg = eeg.cuda()
        print(eeg.shape)
        feature = self.enc(eeg)
        print(feature.shape)
        stop
        return feature, images_mask

    def __repr__(self):
        s = "model: RRG_EEG\n"
        s += "(enc):" + str(self.enc) + '\n'
        s += "(dec):" + str(self.dec) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
