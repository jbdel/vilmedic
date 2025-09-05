import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, AutoModel, AutoModelForCausalLM
from omegaconf.dictconfig import DictConfig
from importlib import import_module
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


class EncoderDecoderHFModel(nn.Module):
    def __init__(self, config, train_dataset, logger):
        super().__init__()       
        # Get encoder and decoder configs
        vision_config = config.get('vision')
        decoder_config = config.get('decoder')
        encoderdecoder_config = config.get('encoderdecoder')
        
        if encoderdecoder_config:
            self.model = VisionEncoderDecoderModel.from_pretrained(encoderdecoder_config)
        else:
            # Create encoder
            if isinstance(vision_config, DictConfig):
                encoder = self._create_model_from_config(vision_config, is_encoder=True)
            elif isinstance(vision_config, str):
                encoder = AutoModel.from_pretrained(vision_config)
            else:
                raise NotImplementedError(f"Unsupported vision config type: {type(vision_config)}")
            
            # Create decoder
            if isinstance(decoder_config, DictConfig):
                decoder = self._create_model_from_config(
                    decoder_config, 
                    is_encoder=False,
                    tokenizer=train_dataset.tokenizer
                )
            elif isinstance(decoder_config, str):
                decoder = AutoModelForCausalLM.from_pretrained(decoder_config, add_cross_attention=True)
            else:
                raise NotImplementedError(f"Unsupported decoder config type: {type(decoder_config)}")
            
            # Create VisionEncoderDecoder model
            self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
            
            # Set special tokens
            tokenizer = train_dataset.tokenizer
            self.model.config.decoder_start_token_id = tokenizer.cls_token_id
            self.model.config.pad_token_id = tokenizer.pad_token_id
            self.model.config.vocab_size = self.model.decoder.config.vocab_size
            
            # Ensure decoder is properly configured
            assert self.model.decoder.config.is_decoder
            assert self.model.decoder.config.add_cross_attention
        
        logger.info(f"Model initialized: {self.__class__.__name__}")
    
    def _create_model_from_config(self, config, is_encoder=True, tokenizer=None):
        """Create model from DictConfig"""
        proto_model = config.pop('proto_model')
        proto_config = config.pop('proto_config')
        
        transformers_module = import_module("transformers")
        config_class = getattr(transformers_module, CONFIG_MAPPING_NAMES[proto_config])
        
        if is_encoder:
            model_class = getattr(transformers_module, MODEL_MAPPING_NAMES[proto_model])
        else:
            model_class = getattr(transformers_module, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[proto_model])
        
        # Get config args
        proto_config_args = config.pop('proto_config_args', {})
        
        # For decoder, set tokenizer-specific configs
        if not is_encoder and tokenizer:
            proto_config_args['vocab_size'] = tokenizer.vocab_size
            proto_config_args['unk_token_id'] = tokenizer.unk_token_id
            proto_config_args['bos_token_id'] = tokenizer.cls_token_id
            proto_config_args['eos_token_id'] = tokenizer.sep_token_id
            proto_config_args['pad_token_id'] = tokenizer.pad_token_id
            proto_config_args['is_decoder'] = True
            proto_config_args['add_cross_attention'] = True
        
        model_config = config_class(**proto_config_args)
        return model_class(model_config)
    
    def forward(self, input_ids, attention_mask, images, images_mask=None, **kwargs):
        """Forward pass handling both single and multi-image inputs"""
        B = images.shape[0]
        device = images.device
        
        # Multi-image case
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            
            # Create mask if not provided
            if images_mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=device)
            else:
                mask = images_mask.to(device).bool()
            
            # Flatten and encode all crops
            flat_pixels = images.view(B * N, C, H, W)
            enc_out = self.model.encoder(pixel_values=flat_pixels)
            flat_hidden = enc_out.last_hidden_state
            S, D = flat_hidden.shape[1], flat_hidden.shape[2]
            
            # Concatenate along sequence axis
            encoder_hidden_states = flat_hidden.view(B, N * S, D)
            
            # Project if needed
            if (self.model.encoder.config.hidden_size != 
                self.model.decoder.config.hidden_size and
                self.model.decoder.config.cross_attention_hidden_size is None):
                encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)
            
            # Build patch-level attention mask
            attn_mask = mask.unsqueeze(-1).expand(B, N, S).reshape(B, N * S).long()
            
            # Decode
            decoder_outputs = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attn_mask,
                labels=input_ids,
            )
        
        # Single-image case
        elif images.dim() == 4:
            enc_out = self.model.encoder(pixel_values=images)
            encoder_hidden_states = enc_out.last_hidden_state
            
            if (self.model.encoder.config.hidden_size != 
                self.model.decoder.config.hidden_size and
                self.model.decoder.config.cross_attention_hidden_size is None):
                encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)
            
            decoder_outputs = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                labels=input_ids,
            )
        else:
            raise ValueError(f"Unexpected images dimension: {images.dim()}")
        
        return decoder_outputs
    
    def generate(self, images, images_mask=None, **kwargs):
        """Generation method for evaluation - handles both single and multi-image"""
        import torch
        from transformers.modeling_outputs import BaseModelOutput
        
        device = images.device
        
        # Multi-image case
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            
            # Create mask if not provided
            if images_mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=device)
            else:
                mask = images_mask.to(device).bool()
            
            # Flatten and encode all crops
            flat_pixels = images.view(B * N, C, H, W)
            enc_out = self.model.encoder(
                pixel_values=flat_pixels,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
            flat_hidden = enc_out.last_hidden_state
            S, D = flat_hidden.shape[1], flat_hidden.shape[2]
            
            # Concatenate along sequence axis
            concat_hidden = flat_hidden.view(B, N * S, D)
            
            # Project if needed
            if (self.model.encoder.config.hidden_size != 
                self.model.decoder.config.hidden_size and
                self.model.decoder.config.cross_attention_hidden_size is None):
                concat_hidden = self.model.enc_to_dec_proj(concat_hidden)
            
            # Build patch-level attention mask
            attn_mask = mask.unsqueeze(-1).expand(B, N, S).reshape(B, N * S).long()
            
            # Generate with precomputed encoder outputs and mask
            encoder_outputs = BaseModelOutput(last_hidden_state=concat_hidden)
            
            return self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attn_mask,
                **kwargs
            )
        
        # Single-image case
        elif images.dim() == 4:
            return self.model.generate(
                pixel_values=images,
                **kwargs
            )
        else:
            raise ValueError(f"Unexpected images dimension: {images.dim()}")
