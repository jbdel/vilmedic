import torch.nn as nn
from ..rnn.utils import get_n_params, set_embeddings
import copy

# v4.3.2
import torch
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from .beam import beam_search
from transformers.models.roberta import modeling_roberta
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions
from ..vision.cnn import CNN


class MultimodalEnc(nn.Module):
    def __init__(self, encoder, cnn):
        super().__init__()
        self.encoder = encoder
        self.cnn = CNN(**cnn)
        self.visual_projection = nn.Linear(cnn.pop("visual_embedding_dim"), self.encoder.config.hidden_size)

    def forward(
            self,
            input_ids,
            images,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.encoder.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.encoder.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.encoder.config.use_return_dict

        if self.encoder.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.encoder.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.encoder.get_head_mask(head_mask, self.encoder.config.num_hidden_layers)

        with torch.no_grad():
            visual_features, _ = self.cnn(images.cuda())

        # Add visual attribute to encoder_outputs
        visual_features = self.visual_projection(visual_features)

        embedding_output = self.encoder.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        embedding_output = torch.cat((embedding_output, visual_features), dim=1)

        # update mask accordingly
        image_mask = torch.ones(
            visual_features.size(-2), device=visual_features.device
        ).expand(visual_features.size()[:-1]).long()
        attention_mask = torch.cat((attention_mask, image_mask), dim=-1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.encoder.get_extended_attention_mask(attention_mask, input_shape,
                                                                                         device)
        encoder_extended_attention_mask = None

        encoder_outputs = self.encoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.encoder.pooler(sequence_output) if self.encoder.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        out = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        out.attention_mask = attention_mask
        return out

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.cnn.train(False)
        return self


class MultimodalEncDec(EncoderDecoderModel):
    def __init__(self, encoder, decoder, cnn, **kwargs):
        enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder.pop('proto'),
                                                                      decoder.pop('proto'))

        super().__init__(encoder=enc_dec.encoder, decoder=enc_dec.decoder, config=enc_dec.config)
        self.encoder = MultimodalEnc(self.encoder, cnn)
        self.to_tile = ["last_hidden_state", "attention_mask"]

    def forward(
            self,
            input_ids=None,
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

        if encoder_outputs is None:
            kwargs_encoder = {argument: value for argument, value in kwargs.items() if
                              not argument.startswith("decoder_")}

            # Encoder
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask

        # Decode
        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
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


class ConcatVisualSeq2SeqHug(nn.Module):
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
