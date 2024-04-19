import torch
import torch.nn as nn
from importlib import import_module

from vilmedic.models.utils import get_n_params
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers import VisionEncoderDecoderModel, AutoModel, BertLMHeadModel
from torch.nn import CrossEntropyLoss

from vilmedic.blocks.huggingface.encoder_decoder.vision_evaluation import evaluation

from omegaconf.dictconfig import DictConfig


class RRG_HF(nn.Module):
    def __init__(self, encoderdecoder=None, decoder=None, vision=None, dl=None, **kwargs):
        super().__init__()
        assert (encoderdecoder is None) ^ (decoder is None or vision is None), \
            "Either proto should be provided, or both decoder and vision should be provided."

        if encoderdecoder is not None:
            self.model = VisionEncoderDecoderModel.from_pretrained(encoderdecoder)
        else:
            # Encoder
            if isinstance(vision, DictConfig):
                assert "proto_model" in vision
                assert "proto_config" in vision
                proto_model = vision.pop('proto_model')
                proto_config = vision.pop('proto_config')
                assert proto_config in CONFIG_MAPPING_NAMES
                assert proto_model in MODEL_MAPPING_NAMES

                transformers_module = import_module("transformers")
                config_class = getattr(transformers_module, CONFIG_MAPPING_NAMES[proto_config])
                model_class = getattr(transformers_module, MODEL_MAPPING_NAMES[proto_model])

                if "proto_config_args" in vision:
                    proto_config_args = vision.pop('proto_config_args')
                else:
                    print("using default config for vision")
                    proto_config_args = {}

                enc_config = config_class(**proto_config_args)
                encoder = model_class(enc_config)

            elif isinstance(vision, str):
                encoder = AutoModel.from_pretrained(vision)
            else:
                raise NotImplementedError(type(vision))

            # decoder
            if isinstance(decoder, DictConfig):
                assert "proto_model" in decoder
                assert "proto_config" in decoder
                proto_model = decoder.pop('proto_model')
                proto_config = decoder.pop('proto_config')
                assert proto_config in CONFIG_MAPPING_NAMES
                assert proto_model in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

                transformers_module = import_module("transformers")
                config_class = getattr(transformers_module, CONFIG_MAPPING_NAMES[proto_config])
                model_class = getattr(transformers_module, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[proto_model])

                if "proto_config_args" in decoder:
                    proto_config_args = decoder.pop('proto_config_args')
                else:
                    print("using default config for decoder")
                    proto_config_args = {}

                if dl:
                    tokenizer = dl.dataset.seq.tokenizer
                    proto_config_args.vocab_size = tokenizer.vocab_size
                    proto_config_args.unk_token_id = tokenizer.unk_token_id
                    proto_config_args.bos_token_id = tokenizer.cls_token_id
                    proto_config_args.eos_token_id = tokenizer.sep_token_id
                    proto_config_args.pad_token_id = tokenizer.pad_token_id

                proto_config_args.is_decoder = True
                proto_config_args.add_cross_attention = True

                dec_config = config_class(**proto_config_args)
                decoder = model_class(dec_config)

            elif isinstance(decoder, str):
                decoder = AutoModel.from_pretrained(decoder)
            else:
                raise NotImplementedError(type(decoder))

            self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

            if dl:
                tokenizer = dl.dataset.seq.tokenizer
                self.model.config.decoder_start_token_id = tokenizer.cls_token_id
                self.model.config.pad_token_id = tokenizer.pad_token_id
                self.model.config.vocab_size = self.model.decoder.config.vocab_size

            assert self.model.decoder.config.is_decoder
            assert self.model.decoder.config.add_cross_attention

            # Evaluation
            self.eval_func = evaluation

    def forward(self, input_ids, attention_mask, images, images_mask=None, epoch=None, iteration=None, **kwargs):
        image_dim = images.dim()
        if image_dim == 5:
            first_image = images[:, 0, :, :, :]
        elif image_dim == 4:
            first_image = images
        else:
            raise NotImplementedError(image_dim)

        encoder_outputs = self.model.encoder(
            pixel_values=first_image.cuda(),
        )

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
                self.model.encoder.config.hidden_size != self.model.decoder.config.hidden_size
                and self.model.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.model.decoder(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            encoder_hidden_states=encoder_hidden_states.cuda(),
            encoder_attention_mask=None,
            labels=input_ids.cuda(),
        )

        out = vars(decoder_outputs)
        return out

    def __repr__(self):
        s = "model: RRG_HF\n"
        s += "(model):" + str(self.model) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
