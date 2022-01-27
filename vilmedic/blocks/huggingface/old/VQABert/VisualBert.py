from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertEmbeddings, \
    BertPredictionHeadTransform
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm


class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.
    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits
        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.
        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)


class BertVisioLinguisticEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(
            self, input_ids, token_type_ids=None
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings

    def encode_image(
            self,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment=None,
    ):

        visual_embeddings = self.projection(visual_embeddings)
        token_type_embeddings_visual = self.token_type_embeddings_visual(
            visual_embeddings_type
        )

        # get position_embeddings
        # this depends on image_text_alignment
        position_embeddings_visual = self.get_position_embeddings_visual(
            visual_embeddings, image_text_alignment=image_text_alignment
        )

        # calculate visual embeddings
        v_embeddings = (
                visual_embeddings
                + position_embeddings_visual
                + token_type_embeddings_visual
        )
        return v_embeddings

    def get_position_embeddings_visual(
            self, visual_embeddings, image_text_alignment=None
    ):

        if image_text_alignment is not None:
            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the
            # image feature. -1 is the padding value.
            image_text_alignment_mask = (
                (image_text_alignment != -1).long().to(image_text_alignment.device)
            )
            # Get rid of the -1.
            image_text_alignment = image_text_alignment_mask * image_text_alignment

            # position_embeddings_visual
            # = Batch x image_length x alignment length x dim
            position_embeddings_visual = self.position_embeddings(
                image_text_alignment
            ) * image_text_alignment_mask.unsqueeze(-1)
            position_embeddings_visual = position_embeddings_visual.sum(2)

            # We want to averge along the alignment_number dimension.
            image_text_alignment_mask = image_text_alignment_mask.sum(2)
            image_text_alignment_mask[image_text_alignment_mask == 0] = torch.tensor(
                [1], dtype=torch.long
            )  # Avoid devide by zero error
            position_embeddings_visual = (
                    position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)
            )

            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )

            position_embeddings_visual = (
                    position_embeddings_visual
                    + self.position_embeddings_visual(position_ids_visual)
            )
        else:
            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )
            position_embeddings_visual = self.position_embeddings_visual(
                position_ids_visual
            )

        return position_embeddings_visual

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            visual_embeddings=None,
            visual_embeddings_type=None,
            image_text_alignment=None,
            **kwargs
    ):
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        text_embeddings = self.encode_text(input_ids, token_type_ids=token_type_ids)

        # visual embeddings
        if visual_embeddings is not None and visual_embeddings_type is not None:
            v_embeddings = self.encode_image(
                visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                image_text_alignment=image_text_alignment,
            )

            # Concate the two:
            embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        else:
            embeddings = text_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBERTBase(nn.Module):
    def __init__(
            self,
            num_labels=None,
            visual_embedding_dim=None,
            **kwargs
    ):
        super().__init__()
        assert visual_embedding_dim is not None
        assert num_labels is not None
        config = BertConfig.from_pretrained('bert-base-uncased',
                                            num_labels=num_labels,
                                            **kwargs)


        self.config = config
        self.config.visual_embedding_dim = visual_embedding_dim

        self.embeddings = BertVisioLinguisticEmbeddings(self.config)

        # self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # to remove
        # self.lstm = nn.LSTM(
        #     input_size=self.config.hidden_size,
        #     hidden_size=self.config.hidden_size,
        #     num_layers=1,
        #     batch_first=True
        # )

        self.proj = nn.Linear(self.config.visual_embedding_dim, config.hidden_size)

        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.config),
            nn.Linear(self.config.hidden_size, num_labels)
        )

        self.init_weights()

        self.loss_fct = nn.CrossEntropyLoss()
        self.num_labels = num_labels

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            input_ids,
            labels,
            input_mask=None,
            attention_mask=None,
            visual_embeddings=None,
            token_type_ids=None,
            visual_embeddings_type=None,
            image_text_alignment=None,
    ):

        image_mask = torch.ones(
            visual_embeddings.size(-2), device=visual_embeddings.device
        ).expand(visual_embeddings.size()[:-1]).long()

        attention_mask = torch.cat((input_mask, image_mask), dim=-1)

        if visual_embeddings_type is None:
            visual_embeddings_type = torch.zeros_like(image_mask)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embeddings = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        # text_embeddings = self.embeddings(input_ids)
        # text_embeddings, _ = self.lstm(text_embeddings)
        # v_embeddings = self.proj(visual_embeddings)
        # embeddings = torch.cat(
        #     (text_embeddings, v_embeddings), dim=1
        # )

        # Only keep last layer hidden states (no output attentions)
        encoded_layers = self.encoder(embeddings, extended_attention_mask)
        sequence_output = encoded_layers[0]

        # pooled_output = self.pooler(sequence_output)
        attn_data_list = encoded_layers[1:]

        # # VQA
        index_to_gather = input_mask.sum(1) - 2
        pooled_output = torch.gather(
            sequence_output,
            1,
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)

        output_dict = {"logits": reshaped_logits,
                       "loss": self.loss_fct(reshaped_logits, labels)}
        return output_dict
