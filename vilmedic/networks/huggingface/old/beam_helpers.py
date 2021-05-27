from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, logging, \
    ModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput

from transformers.generation_beam_search import BeamScorer, BeamHypotheses, UserDict

from transformers.generation_logits_process import (
    LogitsProcessorList,
)
import copy

logger = logging.get_logger(__name__)


def beam_search(
        self,
        encdecs,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        model_kwargs_list=None,
        **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    # !!!! JB !!!
    # **model_kwargs has lost keys from the function signature
    # need to remove the keys from the dict in model_kwargs_list to match current state of **model_kwargs
    ref_keys = model_kwargs.keys()
    for m in model_kwargs_list:
        to_del_k = [k for k in m.keys() if k not in ref_keys]
        for k in to_del_k:
            del m[k]


    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    max_length = max_length if max_length is not None else self.config.max_length
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    assert (
            num_beams * batch_size == batch_beam_size
    ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while cur_len < max_length:
        models_inputs = [encdec.prepare_inputs_for_generation(input_ids, **m)
                         for encdec, m in zip(encdecs, model_kwargs_list)]

        outputs = [encdec(**model_inputs,
                          return_dict=True,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states)
                   for encdec, model_inputs in zip(encdecs, models_inputs)]

        next_token_logits = sum(o.logits[:, -1, :] for o in outputs)

        # adjust tokens for Bart, *e.g.*
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length
        )

        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        cur_len = cur_len + 1

        model_kwargs_list = [
            encdec._update_model_kwargs_for_generation(o, m, is_encoder_decoder=self.config.is_encoder_decoder)
            for encdec, o, m in zip(encdecs, outputs, model_kwargs_list)]

        # model_kwargs = self._update_model_kwargs_for_generation(
        #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        # )
        #
        for m in model_kwargs_list:
            if m["past"] is not None:
                m["past"] = self._reorder_cache(m["past"], beam_idx)

        if beam_scorer.is_done:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    )

    return sequence_outputs["sequences"]


def generate(
        self,
        encdecs,
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
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    # set init values

    num_beams = num_beams if num_beams is not None else self.config.num_beams
    num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    max_length = max_length if max_length is not None else self.config.max_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states

    if model_kwargs.get("attention_mask", None) is None:
        # init `attention_mask` depending on `pad_token_id`
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )

    # special case if pad_token_id is not defined
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    # Storing encoder_input_ids for logits_processor that could use them
    encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

    if self.config.is_encoder_decoder:
        # add encoder_outputs to model_kwargs
        model_kwargs = [encdec._prepare_encoder_decoder_kwargs_for_generation(input_ids, copy.deepcopy(model_kwargs))
                        for encdec in encdecs]

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs[0]:
            input_ids = model_kwargs[0].pop("decoder_input_ids")
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

        if "encoder_outputs" not in model_kwargs[0] or not isinstance(model_kwargs[0]["encoder_outputs"], ModelOutput):
            raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # determine generation mode
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False

    # set model_kwargs
    for m in model_kwargs:
        m["use_cache"] = use_cache

    # get distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=encoder_input_ids,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    )

    if is_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # interleave with `num_beams`
        old_inputs = copy.deepcopy(input_ids)
        old_model_kwargs = copy.deepcopy(model_kwargs)

        # need input_ids only once
        input_ids, _ = self._expand_inputs_for_generation(
            old_inputs, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs[0]
        )

        # need model_kwargs updated for each model
        model_kwargs = [e._expand_inputs_for_generation(old_inputs,
                                                        expand_size=num_beams,
                                                        is_encoder_decoder=e.config.is_encoder_decoder, **m)[1]
                        for m, e in zip(old_model_kwargs, encdecs)]

        return beam_search(
            self,
            encdecs,
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            model_kwargs_list=model_kwargs,
            **model_kwargs[0],
        )
    else:
        raise NotImplementedError


class BeamSearchScorer(BeamScorer):
    def __init__(
            self,
            batch_size: int,
            max_length: int,
            num_beams: int,
            device: torch.device,
            length_penalty: Optional[float] = 1.0,
            do_early_stopping: Optional[bool] = False,
            num_beam_hyps_to_keep: Optional[int] = 1,
            num_beam_groups: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                        len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            final_beam_tokens: torch.LongTensor,
            final_beam_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )
