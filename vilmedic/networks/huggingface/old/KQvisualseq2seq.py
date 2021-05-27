import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

# v4.3.2
import torch
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from .beam import beam_search
from transformers.models.roberta import modeling_roberta
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions
from ..vision.cnn import CNN
from .multimodal_roberta import RobertaModel as MultimodalRoberta
from transformers import RobertaConfig


class MultimodalEncDec(EncoderDecoderModel):
    def __init__(self, encoder, decoder, cnn, **kwargs):
        enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder.pop('proto'),
                                                                      decoder.pop('proto'))

        super().__init__(encoder=enc_dec.encoder, decoder=enc_dec.decoder, config=enc_dec.config)

        config = {'config': RobertaConfig(**{
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "transformers_version": "4.5.1",
            "type_vocab_size": 1,
            "use_cache": True,
            "vocab_size": 50265
        })}
        self.encoder: MultimodalRoberta = MultimodalRoberta.from_pretrained(
            'data/report_sum/huggingface/biomed_roberta_base', **config)
        self.cnn = CNN(**cnn)
        self.visual_projection = nn.Linear(cnn.pop("visual_embedding_dim"), self.encoder.config.hidden_size)

    def forward(
            self,
            input_ids=None,
            images=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            with torch.no_grad():
                visual_features, _ = self.cnn(images.cuda())

            # Add visual attribute to encoder_outputs
            visual_features = self.visual_projection(visual_features)

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                visual_features=visual_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.cnn.train(False)
        return self


class KQVisualSeq2SeqHug(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()

        self.enc_dec = MultimodalEncDec(encoder, decoder, **kwargs)
        self.enc = self.enc_dec.encoder
        self.dec = self.enc_dec.decoder

        self.bos_token_id = self.dec.config.bos_token_id
        self.eos_token_id = self.dec.config.eos_token_id
        self.pad_token_id = self.dec.config.pad_token_id

        # Evaluation
        self.eval_func = beam_search

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        input_ids = input_ids.cuda()
        decoder_input_ids = decoder_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        decoder_attention_mask = decoder_attention_mask.cuda()

        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=decoder_input_ids,
                           **kwargs)
        out = vars(out)
        return out

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = str(type(self.enc_dec.encoder).__name__) + '(' + str(self.enc_dec.encoder.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
