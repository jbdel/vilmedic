import torch.nn as nn

import torch
from vilmedic.networks.blocks.huggingface.encoder_decoder.evaluation import evaluation
from vilmedic.networks.models.utils import get_n_params

# v4.3.2
from transformers.modeling_outputs import Seq2SeqLMOutput
from vilmedic.networks.blocks.huggingface.encoder_decoder.encoder_decoder_model import EncoderDecoderModel


class MultimodalEnc(nn.Module):
    def __init__(self, encoder, cnn):
        super().__init__()
        self.encoder = encoder

        cnn_func = cnn.pop('proto')
        self.visual_projection = nn.Linear(cnn.pop("visual_embedding_dim"), self.encoder.config.hidden_size)
        self.cnn = eval(cnn_func)(**cnn)

    def forward(self, input_ids, images, **kwargs):
        # Encoder
        encoder_outputs = self.encoder(input_ids, **kwargs)

        # CNN
        with torch.no_grad():
            visual_features = self.cnn(images.cuda())

        # Add visual attribute to encoder_outputs
        visual_features = self.visual_projection(visual_features)
        encoder_outputs.visual_features = visual_features

        return encoder_outputs

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

        # beam param
        self.to_tile = ["last_hidden_state", "visual_features"]

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

        visual_features = encoder_outputs.visual_features
        # Concat modalities
        encoder_hidden_states = torch.cat((encoder_outputs.last_hidden_state, visual_features), dim=1)

        # update mask accordingly
        image_mask = torch.ones(
            visual_features.size(-2), device=visual_features.device
        ).expand(visual_features.size()[:-1]).long()
        attention_mask = torch.cat((attention_mask, image_mask), dim=-1)

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


class SumHugMulti(nn.Module):

    def __init__(self, encoder, decoder, cnn, **kwargs):
        super().__init__()

        self.cnn = eval(cnn.pop('proto'))(**cnn)
        self.visual_projection = nn.Linear(cnn.pop("visual_embedding_dim"), self.encoder.config.hidden_size)

        self.enc_dec = EncoderDecoderModel(encoder, decoder)
        self.enc = self.enc_dec.enc

        # Evaluation
        self.eval_func = evaluation
        self.enc_dec.to_tile = ["last_hidden_state", "visual_features"]

        self.bos_token_id = self.enc_dec.dec.config.bos_token_id
        self.eos_token_id = self.enc_dec.dec.config.eos_token_id
        self.pad_token_id = self.enc_dec.dec.config.pad_token_id


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

        visual_features = encoder_outputs.visual_features
        # Concat modalities
        encoder_hidden_states = torch.cat((encoder_outputs.last_hidden_state, visual_features), dim=1)

        # update mask accordingly
        image_mask = torch.ones(
            visual_features.size(-2), device=visual_features.device
        ).expand(visual_features.size()[:-1]).long()
        attention_mask = torch.cat((attention_mask, image_mask), dim=-1)

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

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = str(type(self.enc_dec.encoder).__name__) + '(' + str(self.enc_dec.encoder.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
