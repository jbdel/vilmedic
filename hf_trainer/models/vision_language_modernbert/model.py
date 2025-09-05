import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers import AutoBackbone
from transformers import ModernBertDecoderForCausalLM, ModernBertDecoderConfig, AutoTokenizer
from transformers import PreTrainedModel, PretrainedConfig


class VisionLanguageConfig(PretrainedConfig):
    model_type = "vision_language_modernbert"

    def __init__(
        self,
        vision_model_name: str = "IAMJB/maira-2-dinov2",
        text_model_config: Optional[PretrainedConfig] = None,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        # Auto-populate ModernBert decoder config if not provided (align with maira2-defn.py)
        if text_model_config is None:
            _ = AutoTokenizer.from_pretrained(tokenizer_name)
            self.text_config = ModernBertDecoderConfig.from_pretrained("jhu-clsp/ettin-decoder-17m")
        else:
            self.text_config = text_model_config
        self.tokenizer_name = tokenizer_name
        # Token ids are stored here for convenience by the wrapper
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class VisionLanguageModel(PreTrainedModel):
    config_class = VisionLanguageConfig

    def __init__(self, config: VisionLanguageConfig):
        super().__init__(config)
        self.config = config

        # Vision encoder
        self.vision_encoder = AutoBackbone.from_pretrained(config.vision_model_name)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Text decoder
        assert config.text_config is not None, "text_model_config must be provided"
        self.text_decoder = ModernBertDecoderForCausalLM(config.text_config)

        # Determine in/out feature sizes for projection
        in_features = getattr(self.vision_encoder.config, 'hidden_size', None)
        if in_features is None:
            hidden_sizes = getattr(self.vision_encoder.config, 'hidden_sizes', None)
            if isinstance(hidden_sizes, (list, tuple)) and len(hidden_sizes) > 0:
                in_features = int(hidden_sizes[-1])
        if in_features is None:
            raise ValueError("Unable to infer vision hidden size from backbone config")

        out_features = int(getattr(self.text_decoder.config, 'hidden_size', None))
        if out_features is None:
            raise ValueError("Unable to infer text hidden size from decoder config")

        # Projection and image marker
        self.vision_projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )
        self.image_token_embedding = nn.Parameter(
            torch.randn(1, 1, out_features) * 0.02
        )

        self.post_init()

    # ---- helpers ----
    def _encode_single_image(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Backbones are frozen; do not build a graph for them.
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values)  # feature_maps returned by BackboneOutput

        feat = outputs.feature_maps[-1]  # Expect [B,C,H,W] or [B,S,C] / [B,C,S]
        if feat.dim() == 4:
            # [B, C, H, W] -> [B, S, C]
            b, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2).contiguous()
        elif feat.dim() == 3:
            # Try to ensure last dim is the channel/hidden dim (== in_features)
            if feat.size(-1) != getattr(self.vision_encoder.config, 'hidden_size', feat.size(-1)):
                # likely [B, C, S] -> [B, S, C]
                feat = feat.transpose(1, 2).contiguous()
            # Optionally drop a ViT-style CLS token if present
            if getattr(self.vision_encoder.config, 'use_cls_token', True) and feat.size(1) > 1:
                feat = feat[:, 1:, :]
        else:
            raise ValueError(f"Unexpected feature map shape from backbone: {feat.shape}")

        projected = self.vision_projection(feat)  # [B, S, Dh]
        bsz = projected.size(0)
        marker = self.image_token_embedding.expand(bsz, 1, -1)  # [B, 1, Dh]
        return torch.cat([marker, projected], dim=1).contiguous()  # [B, S+1, Dh]

    # ---- huggingface forward ----
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = None
        if pixel_values is not None:
            if pixel_values.dim() == 5:
                b, n, c, h, w = pixel_values.shape
                if n > 2:
                    raise ValueError(f"This model supports at most two images per sample (got N={n}).")
                flat_pixels = pixel_values.reshape(b * n, c, h, w)
                img_embeds = self._encode_single_image(flat_pixels)  # [B*N, S+1, Dh]
                s_plus_1 = img_embeds.size(1)
                img_embeds = img_embeds.reshape(b, n, s_plus_1, -1)

                # Build image attention mask
                if images_mask is None:
                    mask_bn = torch.ones((b, n), dtype=torch.bool, device=img_embeds.device)
                else:
                    mask_bn = images_mask.to(img_embeds.device).bool()
                mask_tokens = mask_bn.unsqueeze(-1).expand(b, n, s_plus_1)  # [B, N, S+1]
                image_attention = mask_tokens.reshape(b, n * s_plus_1).long()  # [B, N*(S+1)]

                # Zero-out embeddings of masked images
                img_embeds = img_embeds * mask_tokens.unsqueeze(-1)

                img_embeds = img_embeds.reshape(b, n * s_plus_1, -1)
            elif pixel_values.dim() == 4:
                img_embeds = self._encode_single_image(pixel_values)  # [B, S+1, Dh]
                image_attention = torch.ones(img_embeds.size(0), img_embeds.size(1), dtype=torch.long, device=img_embeds.device)
            else:
                raise ValueError(f"Unexpected pixel_values dimension: {pixel_values.dim()}")

            if input_ids is not None:
                # Build text embeddings via the model's input embedding module
                text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)

                # Extend attention mask
                if attention_mask is not None:
                    if attention_mask.dtype != image_attention.dtype:
                        image_attention = image_attention.to(dtype=attention_mask.dtype)
                    attention_mask = torch.cat([image_attention, attention_mask], dim=1)
                else:
                    attention_mask = image_attention

                # Extend labels to ignore image prefix tokens in loss computation
                if labels is not None:
                    prefix_len = img_embeds.size(1)
                    ignore_index = -100
                    prefix_ignore = torch.full(
                        (labels.size(0), prefix_len),
                        fill_value=ignore_index,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([prefix_ignore, labels], dim=1)

                # Use inputs_embeds path for decoding
                input_ids = None
            else:
                inputs_embeds = img_embeds
                attention_mask = image_attention

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def generate(self, pixel_values=None, images_mask: Optional[torch.Tensor] = None, **generate_kwargs):
        if pixel_values is not None:
            if pixel_values.dim() == 5:
                b, n, c, h, w = pixel_values.shape
                if n > 2:
                    raise ValueError(f"This model supports at most two images per sample (got N={n}).")
                flat_pixels = pixel_values.reshape(b * n, c, h, w)
                img_embeds = self._encode_single_image(flat_pixels)  # [B*N, S+1, Dh]
                s_plus_1 = img_embeds.size(1)
                img_embeds = img_embeds.reshape(b, n, s_plus_1, -1)

                if images_mask is None:
                    mask_bn = torch.ones((b, n), dtype=torch.bool, device=img_embeds.device)
                else:
                    mask_bn = images_mask.to(img_embeds.device).bool()
                mask_tokens = mask_bn.unsqueeze(-1).expand(b, n, s_plus_1)
                attention_mask = mask_tokens.reshape(b, n * s_plus_1).long()
                img_embeds = img_embeds * mask_tokens.unsqueeze(-1)
                img_embeds = img_embeds.reshape(b, n * s_plus_1, -1)
            elif pixel_values.dim() == 4:
                img_embeds = self._encode_single_image(pixel_values)  # [B, S+1, Dh]
                attention_mask = torch.ones(img_embeds.size(0), img_embeds.size(1), dtype=torch.long, device=img_embeds.device)
            else:
                raise ValueError(f"Unexpected pixel_values dimension: {pixel_values.dim()}")

            return self.text_decoder.generate(
                inputs_embeds=img_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        else:
            return self.text_decoder.generate(**generate_kwargs)


class VisionLanguageModernBertModel(nn.Module):
    """
    Thin wrapper around the HuggingFace PreTrainedModel (VisionLanguageModel).
    Exposes the expected forward/generate signatures used by our Trainer.
    """

    def __init__(self, config, train_dataset, logger):
        super().__init__()

        tokenizer = train_dataset.tokenizer

        # Build decoder config with overrides
        decoder_config = ModernBertDecoderConfig.from_pretrained(
            config.get('decoder_config_name', 'jhu-clsp/ettin-decoder-17m')
        )
        decoder_config.vocab_size = tokenizer.vocab_size
        decoder_config.pad_token_id = tokenizer.pad_token_id
        decoder_config.bos_token_id = getattr(tokenizer, 'cls_token_id', None)
        decoder_config.eos_token_id = getattr(tokenizer, 'sep_token_id', None)
        # Optional overrides
        for k, v in dict(config.get('decoder_config_args', {})).items():
            setattr(decoder_config, k, v)

        # Build HF VLM model
        hf_config = VisionLanguageConfig(
            vision_model_name=config.get('vision_model_name', 'IAMJB/maira-2-dinov2'),
            text_model_config=decoder_config,
            tokenizer_name=config.get('tokenizer_name', 'answerdotai/ModernBERT-base'),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=getattr(tokenizer, 'cls_token_id', None),
            eos_token_id=getattr(tokenizer, 'sep_token_id', None),
        )
        self.model = VisionLanguageModel(hf_config)

        # Expose logger
        self.logger = logger
        logger.info(f"Model initialized: {self.__class__.__name__}")

    def forward(self, input_ids, attention_mask, images, images_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            images_mask=images_mask,
            labels=input_ids,
        )

    def generate(self, images, images_mask=None, **kwargs):
        return self.model.generate(pixel_values=images, images_mask=images_mask, **kwargs)