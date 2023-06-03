import torch
from torch import Tensor
from torch.nn import functional as F
import logging
import math
import numpy as np
from typing import Iterable, List, Optional, Tuple

from question_generation.topK import topk_huggingface, ConstrainedHypothesis

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    constraints: Optional[List[Optional[ConstrainedHypothesis]]] = None,
    prune_factor: Optional[int] = None,
    sat_tolerance: Optional[int] = None,
    look_ahead_step: Optional[int] = None,
    look_ahead_width: Optional[int] = None,
    alpha: Optional[float] = None,
    fusion_t: Optional[float] = None,
    look_ahead_sample: Optional[bool] = None,
    **model_specific_kwargs
) -> torch.LongTensor:
    r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Adapted in part from `Facebook's XLM beam search code`_.

    .. _`Facebook's XLM beam search code`:
       https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


    Parameters:

        input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
            The sequence used as a prompt for the generation. If `None` the method initializes
            it as an empty `torch.LongTensor` of shape `(1,)`.

        max_length: (`optional`) int
            The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

        min_length: (`optional`) int
            The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

        do_sample: (`optional`) bool
            If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        early_stopping: (`optional`) bool
            if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        num_beams: (`optional`) int
            Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        repetition_penalty: (`optional`) float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

        pad_token_id: (`optional`) int
            Padding token. Default to specicic model pad_token_id or None if it does not exist.

        bos_token_id: (`optional`) int
            BOS token. Defaults to `bos_token_id` as defined in the models config.

        eos_token_id: (`optional`) int
            EOS token. Defaults to `eos_token_id` as defined in the models config.

        length_penalty: (`optional`) float
            Exponential penalty to the length. Default to 1.

        no_repeat_ngram_size: (`optional`) int
            If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
        bad_words_ids: (`optional`) list of lists of int
            `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

        num_return_sequences: (`optional`) int
            The number of independently computed returned sequences for each element in the batch. Default to 1.

        attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            Defaults to `None`.

            `What are attention masks? <../glossary.html#attention-mask>`__

        decoder_start_token_id=None: (`optional`) int
            If an encoder-decoder model starts decoding with a different token than BOS.
            Defaults to `None` and is changed to `BOS` later.

        use_cache: (`optional`) bool
            If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

        model_specific_kwargs: (`optional`) dict
            Additional model specific kwargs will be forwarded to the `forward` function of the model.

    Return:

        output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
            sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

    """

    # We cannot generate if the model does not have a LM head
    if self.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
        bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    if hasattr(self.config, "vocab_size"):
        vocab_size = self.config.vocab_size
    elif (
        self.config.is_encoder_decoder
        and hasattr(self.config, "decoder")
        and hasattr(self.config.decoder, "vocab_size")
    ):
        vocab_size = self.config.decoder.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if self.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if self.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    if num_beams > 1:
        output = _generate_beam_search(
            self,
            input_ids=input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            vocab_size=vocab_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            constraints=constraints,
            prune_factor=prune_factor,
            sat_tolerance=sat_tolerance,
            look_ahead_step=look_ahead_step,
            look_ahead_width=look_ahead_width,
            alpha=alpha,
            fusion_t=fusion_t,
            look_ahead_sample=look_ahead_sample,
            model_specific_kwargs=model_specific_kwargs,
        )
    else:
        raise NotImplementedError
    return output


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams * 2
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, num_met):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        #score = sum_logprobs / math.pow((5 + len(hyp) + 1) / 6.0, self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, num_met))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            #cur_score = best_sum_logprobs / math.pow((5 + cur_len + 1) / 6.0, self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


# from transformers import AutoTokenizer, AutoModelWithLMHead
# tokenizer = AutoTokenizer.from_pretrained('gpt2')


def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        constraints,
        prune_factor,
        sat_tolerance,
        look_ahead_step,
        look_ahead_width,
        alpha,
        fusion_t,
        look_ahead_sample,
        model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """
    # end condition
    cons_eos = constraints[0].eos()

    last_non_masked_idx = (torch.sum(attention_mask, dim=1) - 1).int()
    start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, self.config.vocab_size).unsqueeze(1).long()

    init_length = cur_len
    position_ids = torch.tensor([list(range(init_length)) for i in range(input_ids.shape[0])])
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
    position_ids = position_ids.to(input_ids.device)

    decode_kwargs = {'max_length': max_length, 'min_length': min_length, 'do_sample': do_sample,
                     'repetition_penalty': repetition_penalty, 'no_repeat_ngram_size': no_repeat_ngram_size,
                     'bad_words_ids': bad_words_ids, 'pad_token_id': pad_token_id, 'eos_token_id': eos_token_id,
                     'num_return_sequences': num_return_sequences, 'vocab_size': vocab_size, 'use_cache': use_cache,
                     'model_specific_kwargs': model_specific_kwargs}

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None

    # done sentences
    done = [False for _ in range(batch_size)]

    # init number of met clauses
    num_mets = [x.num_met() for x in constraints]

    # history of token score
    score_history = None

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1) if past else position_ids

        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        if cur_len == init_length:
            next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
        else:
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]
        if self.config.is_encoder_decoder and do_sample is False:
            # TODO (PVP) still a bit hacky here - there might be a better solution
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        scores = self.postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        if do_sample:
            raise NotImplementedError
        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            full_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(full_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            # prepare look ahead input
            assert self._use_cache(outputs, use_cache) and use_cache, 'model not using past'
            temp_attention_mask = attention_mask if self.config.is_encoder_decoder else \
                torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            temp_position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

            pick_scores, pick_tokens, constraints, num_mets = topk_huggingface(timestep=cur_len,
                                                                               batch_size=batch_size,
                                                                               beam_size=num_beams,
                                                                               vocab_size=vocab_size,
                                                                               pad_token_id=pad_token_id,
                                                                               prune_factor=prune_factor,
                                                                               sat_tolerance=sat_tolerance,
                                                                               inactive=np.zeros((batch_size, num_beams)),
                                                                               scores=full_scores,
                                                                               hypotheses=constraints,
                                                                               num_fill=2 * num_beams,
                                                                               look_ahead_step=look_ahead_step,
                                                                               look_ahead_width=look_ahead_width,
                                                                               alpha=alpha,
                                                                               fusion_t=fusion_t,
                                                                               look_ahead_sample=look_ahead_sample,
                                                                               init_length=init_length,
                                                                               max_length=max_length,
                                                                               model=self,
                                                                               temp_input_ids=input_ids,
                                                                               temp_attention_mask=temp_attention_mask,
                                                                               temp_position_ids=temp_position_ids,
                                                                               temp_past=past,
                                                                               decode_kwargs=decode_kwargs,
                                                                               model_specific_kwargs=model_specific_kwargs)

            next_scores = torch.tensor(pick_scores, dtype=next_scores.dtype, device=next_scores.device)
            next_tokens = torch.tensor(pick_tokens, dtype=next_tokens.dtype, device=next_tokens.device)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0, None, -1)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score, constraint, num_met) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], constraints[batch_idx], num_mets[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                sentence_end = token_id.item() in constraint.eos()
                # add to generated hypotheses if end of sentence or last iteration
                if ((eos_token_id is not None) and (token_id.item() == eos_token_id)) or sentence_end:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        torch.cat((input_ids[effective_beam_id], token_id.view([1]))), beam_token_score.item(), num_met,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id, constraint, num_met))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx][:beam_token_rank + 1].max().item(), cur_len=cur_len
            ) or not next_sent_beam

            if len(next_sent_beam) < num_beams:
                if next_sent_beam:
                    pad_candidate = next_sent_beam[-1]
                elif done[batch_idx]:
                    pad_candidate = (0, pad_token_id, 0, None, -1)
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise ValueError('impossible search state')
                next_sent_beam += [pad_candidate] * (num_beams - len(next_sent_beam))

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        constraints = [x[3] for x in next_batch_beam]
        num_mets = [x[4] for x in next_batch_beam]

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        position_ids = position_ids[beam_idx, :]
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cur_len = cur_len + 1

        token_probs = torch.reshape(scores, [batch_size * num_beams, -1])[beam_idx, :]
        token_score = torch.gather(token_probs, -1, beam_tokens[..., None].expand(token_probs.shape))[:, 0]

        if score_history is None:
            score_history = token_score[..., None]
        else:
            previous_history = score_history[beam_idx, :]
            score_history = torch.cat([previous_history, token_score[..., None]], dim=-1)
            score_history *= (beam_scores != 0)[:, None]

        # re-order internal states
        if past is not None:
            past = self._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() not in cons_eos for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            final_num_met = num_mets[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score, final_num_met)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best, best_scores, best_sum_logprobs = [], [], []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        #hyps = sorted(hypotheses.beams, key=lambda x: x[0], reverse=True)[:5]
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: (x[2], x[0]), reverse=True)
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_score, best_hyp, _ = sorted_hyps[0]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
            best_scores.append(best_score)
            best_sum_logprobs.append(best_score * (len(best_hyp) ** length_penalty))

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

    return decoded, best_scores, best_sum_logprobs


@staticmethod
def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


def _use_cache(self, outputs, use_cache):
    """During generation, decide whether to pass the `past` variable to the next forward pass."""
    if len(outputs) <= 1 or use_cache is False:
        return False
    if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
        return False
    return True


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens
