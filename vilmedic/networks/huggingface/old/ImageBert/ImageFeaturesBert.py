from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch.nn as nn
import torch

BertLayerNorm = torch.nn.LayerNorm


class VisioEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, **kwargs):
        # Inputs_ids are of size bs x spatial x feat_dim
        embeddings = input_ids
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ImageFeaturesBert(BertPreTrainedModel):
    def __init__(self, **kwargs):
        config = BertConfig()
        super().__init__(config)

        self.config = config
        self.embeddings = VisioEmbeddings(config)
        config.num_hidden_layers=1
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        input_shape = input_ids.size()
        batch_size, seq_length, _ = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # embedding_output = self.embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     inputs_embeds=inputs_embeds,
        #     past_key_values_length=past_key_values_length,
        # )
        #
        # encoder_outputs = self.encoder(
        #     embedding_output,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_extended_attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # print(encoder_outputs)
        sequence_output = input_ids
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


# if __name__ == '__main__':
#     m = ImageFeaturesBert(visual_embedding_dim=60)
#     x = torch.ones((2, 10, 60))
#     print(m(x))
