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
        s = "model: RRG_HF\n"  # keep
        try:
            enc = getattr(self.model, "encoder", None)
            dec = getattr(self.model, "decoder", None)
            cfg = getattr(self.model, "config", None)

            # ----- Helpers -----
            def _cfg(o): 
                return getattr(o, "config", None) if o is not None else None

            def _get(o, name, default=None):
                return getattr(o, name, default) if o is not None else default

            def _maybe_kv(name, val):
                return f"{name}={val}" if val is not None else None

            # ----- Encoder summary -----
            enc_cfg = _cfg(enc)
            enc_bits = [
                enc.__class__.__name__ if enc is not None else "None",
                _maybe_kv("hidden", _get(enc_cfg, "hidden_size")),
                _maybe_kv("layers", _get(enc_cfg, "num_hidden_layers", _get(enc_cfg, "num_layers"))),
                _maybe_kv("heads", _get(enc_cfg, "num_attention_heads")),
                _maybe_kv("image", _get(enc_cfg, "image_size")),
                _maybe_kv("patch", _get(enc_cfg, "patch_size")),
            ]
            enc_bits = [b for b in enc_bits if b is not None]
            if enc is not None:
                enc_bits.append(f"params={get_n_params(enc)}")

            # ----- Decoder summary -----
            dec_cfg = _cfg(dec)
            dec_bits = [
                dec.__class__.__name__ if dec is not None else "None",
                _maybe_kv("hidden", _get(dec_cfg, "hidden_size")),
                _maybe_kv("layers", _get(dec_cfg, "num_hidden_layers", _get(dec_cfg, "num_layers"))),
                _maybe_kv("heads", _get(dec_cfg, "num_attention_heads")),
                _maybe_kv("vocab", _get(dec_cfg, "vocab_size")),
                _maybe_kv("is_decoder", _get(dec_cfg, "is_decoder")),
                _maybe_kv("cross_attn", _get(dec_cfg, "add_cross_attention")),
            ]
            dec_bits = [b for b in dec_bits if b is not None]
            if dec is not None:
                dec_bits.append(f"params={get_n_params(dec)}")

            # ----- Tokens / config -----
            token_bits = [
                _maybe_kv("pad_id", _get(cfg, "pad_token_id")),
                _maybe_kv("dec_start_id", _get(cfg, "decoder_start_token_id")),
                _maybe_kv("bos_id", _get(dec_cfg, "bos_token_id")),
                _maybe_kv("eos_id", _get(dec_cfg, "eos_token_id")),
                _maybe_kv("unk_id", _get(dec_cfg, "unk_token_id")),
                _maybe_kv("tie_word_emb", _get(cfg, "tie_word_embeddings")),
            ]
            token_bits = [b for b in token_bits if b is not None]

            # ----- Multi-image evaluation hint -----
            eval_mode = None
            if hasattr(self, "eval_func"):
                try:
                    # If set in __init__, evaluation_multi means multi-image batches supported
                    from vilmedic.blocks.huggingface.encoder_decoder.vision_multi_evaluation import evaluation as _eval_multi
                    eval_mode = "multi-image" if self.eval_func is _eval_multi else "single-image"
                except Exception:
                    eval_mode = None

            # ----- Compose pretty print -----
            s += "components:\n"
            s += f"  Encoder: " + " | ".join(enc_bits) + "\n"
            s += f"  Decoder: " + " | ".join(dec_bits) + "\n"
            if token_bits:
                s += "tokens/config:\n"
                s += "  " + ", ".join(token_bits) + "\n"
            if eval_mode:
                s += f"evaluation: {eval_mode}\n"

        except Exception as e:
            s += f"(summary unavailable: {e})\n"

        s += "{}\n".format(get_n_params(self))  # keep
        return s
