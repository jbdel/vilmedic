import torch.nn as nn
import torch
from collections import OrderedDict

from vilmedic.blocks.huggingface.encoder_decoder.evaluation import evaluation
from vilmedic.blocks.huggingface.encoder_decoder.encoder_decoder_model import EncoderDecoderModel


class RRS(nn.Module):

    def __init__(self, encoder, decoder, dl, **kwargs):
        super().__init__()
        self.enc_dec = EncoderDecoderModel(encoder, decoder)

        # Evaluation
        self.eval_func = evaluation

        # Tokenizer
        self.tokenizer = dl.dataset.tgt_tokenizer

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           )

        # print(decoder_input_ids[1])
        # print(decoder_input_ids[1].shape)
        # print(torch.argmax(out["logits"][1], dim=-1).shape)
        # print(self.tokenizer_ss.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        # hyps = self.enc_dec.enc_dec.generate(input_ids.cuda(),
        #                                      attention_mask=attention_mask.cuda(),
        #                                      max_length=80,
        #                                      )
        # # print(hyps)
        # print("####", self.tokenizer.decode(hyps[1], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        # print("####", self.tokenizer.decode(torch.argmax(out["logits"][1], dim=-1), skip_special_tokens=True, clean_up_tokenization_spaces=False))
        # print("####", self.tokenizer.decode(decoder_input_ids[1], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        # print("####")
        return out

    def __repr__(self):
        return "RRS\n" + str(self.enc_dec)
