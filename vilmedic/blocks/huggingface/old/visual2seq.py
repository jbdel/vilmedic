import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

# v4.3.2
from transformers import EncoderDecoderModel
from transformers import AutoModelForCausalLM
from transformers import AutoConfig, EncoderDecoderConfig
from .ImageBert.ImageFeaturesBert import ImageFeaturesBert
from .ImageBert.beam import beam_search
from ..vision.cnn import CNN


class Visual2SeqHug(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, cnn, encoder, decoder, **kwargs):
        super().__init__()

        # Encoder
        enc = ImageFeaturesBert()
        self.backbone = CNN(**cnn)
        self.visual_projection = nn.Linear(encoder.pop("visual_embedding_dim"), enc.config.hidden_size)

        # Decoder
        decoder_pretrained_model_name_or_path = decoder.pop('proto')
        decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
        if decoder_config.is_decoder is False:
            print(
                f"Initializing {decoder_pretrained_model_name_or_path} "
                f"as a decoder model. Cross attention layers are added to "
                f"{decoder_pretrained_model_name_or_path} "
                f"and randomly initialized if "
                f"{decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True

        kwargs_decoder = {"config": decoder_config}

        dec = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(enc.config, dec.config)
        self.enc_dec = EncoderDecoderModel(encoder=enc, decoder=dec, config=config)

        self.enc = self.enc_dec.encoder
        self.dec = self.enc_dec.decoder

        # Tokens
        self.bos_token_id = self.dec.config.bos_token_id
        self.eos_token_id = self.dec.config.eos_token_id
        self.pad_token_id = self.dec.config.pad_token_id

        # Evaluation
        self.eval_func = beam_search
        self.generate_config = {}

        # Embeddings
        if 'src_emb' in kwargs:
            set_embeddings(kwargs['src_emb'], self.enc.embeddings.word_embeddings)
        if 'tgt_emb' in kwargs:
            set_embeddings(kwargs['tgt_emb'], self.dec.bert.embeddings.word_embeddings)

    def forward(self, input_ids, decoder_input_ids, decoder_attention_mask):
        input_ids = input_ids.cuda()
        input_ids, _ = self.backbone(input_ids)
        input_ids = input_ids.detach()
        input_ids = self.visual_projection(input_ids)

        decoder_input_ids = decoder_input_ids.cuda()
        decoder_attention_mask = decoder_attention_mask.cuda()
        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=None,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=decoder_input_ids)
        out = vars(out)
        return out

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.backbone.train(False)
        return self


    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = str(type(self.enc_dec.encoder).__name__) + '(' + str(self.enc_dec.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
