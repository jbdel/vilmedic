import torch
import torch.nn as nn
from importlib import import_module

from vilmedic.models.utils import get_n_params
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers import VisionEncoderDecoderModel, AutoModel, BertLMHeadModel, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

from vilmedic.blocks.huggingface.encoder_decoder.vision_evaluation import evaluation
from vilmedic.blocks.huggingface.encoder_decoder.vision_multi_evaluation import evaluation as evaluation_multi

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
                decoder = AutoModelForCausalLM.from_pretrained(decoder, add_cross_attention=True)
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
            self.eval_func = evaluation_multi



    def forward(self, input_ids, attention_mask, images, images_mask=None, epoch=None, iteration=None, **kwargs):
        """
        Supports both single-image (4D) and multi-image (5D) batches.
        Uses images_mask ([B, N]) to build a patch-level encoder_attention_mask.
        """
        B = images.shape[0]
        device = images.device

        # --- MULTI-IMAGE CASE ---
        if images.dim() == 5:
            # images: [B, N, C, H, W]
            B, N, C, H, W = images.shape

            # if no mask provided, assume all crops valid
            if images_mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=device)
            else:
                mask = images_mask.to(device).bool()

            # 1) flatten and encode all crops at once
            flat_pixels = images.view(B * N, C, H, W).cuda()
            enc_out = self.model.encoder(pixel_values=flat_pixels)
            flat_hidden = enc_out.last_hidden_state                      # [B*N, S, D]
            S, D = flat_hidden.shape[1], flat_hidden.shape[2]

            # 2) concat along sequence axis → [B, N*S, D]
            encoder_hidden_states = flat_hidden.view(B, N * S, D)

            # 3) optional proj to decoder dim
            if (self.model.encoder.config.hidden_size != 
                self.model.decoder.config.hidden_size
                and self.model.decoder.config.cross_attention_hidden_size is None):
                encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

            # 4) build patch-level attention mask [B, N] → [B, N*S]
            attn_mask = mask.unsqueeze(-1).expand(B, N, S).reshape(B, N * S).long()

            # 5) decode with precomputed states + mask
            decoder_outputs = self.model.decoder(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                encoder_hidden_states=encoder_hidden_states.cuda(),
                encoder_attention_mask=attn_mask.cuda(),
                labels=input_ids.cuda(),
            )

        # --- SINGLE-IMAGE CASE (unchanged) ---
        elif images.dim() == 4:
            # images: [B, C, H, W]
            enc_out = self.model.encoder(pixel_values=images.cuda())
            encoder_hidden_states = enc_out.last_hidden_state

            if (
                self.model.encoder.config.hidden_size != self.model.decoder.config.hidden_size
                and self.model.decoder.config.cross_attention_hidden_size is None
            ):
                encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

            decoder_outputs = self.model.decoder(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                encoder_hidden_states=encoder_hidden_states.cuda(),
                encoder_attention_mask=None,
                labels=input_ids.cuda(),
            )

        else:
            raise NotImplementedError(f"Unexpected images.dim() = {images.dim()}")

        return vars(decoder_outputs)


    def __repr__(self):
        s = "model: RRG_HF\n"
        s += "(model):" + str(self.model) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
