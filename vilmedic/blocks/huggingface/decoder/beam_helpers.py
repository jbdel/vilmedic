from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, logging, \
    ModelOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput, BeamSearchScorer

from transformers.generation_beam_search import BeamScorer, BeamHypotheses, UserDict

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers.generation_logits_process import (
    LogitsProcessorList,
)
import copy

logger = logging.get_logger(__name__)


def prepare_inputs_for_generation(
        input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
):
    if past is not None:
        input_ids = input_ids[:, -1:]

    input_dict = {
        "attention_mask": attention_mask,
        "decoder_attention_mask": input_ids.new_ones(input_ids.shape),
        "decoder_input_ids": input_ids,
        "encoder_outputs": encoder_outputs,
        "past_key_values": past,
        "use_cache": use_cache,
    }
    return input_dict


def beam_search(
        hugg_model,
        hugg_models,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        model_kwargs_list=None
) -> Union[BeamSearchOutput, torch.LongTensor]:
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
        models_inputs = [prepare_inputs_for_generation(input_ids, **m)
                         for hm, m in zip(hugg_models, model_kwargs_list)]

        # print(models_inputs)
        # We need to rename the input keys if we give that directly to a decoder
        if hugg_model.config.is_decoder:
            for model_inputs in models_inputs:
                model_inputs["input_ids"] = model_inputs.pop("decoder_input_ids")
                model_inputs["encoder_attention_mask"] = model_inputs.pop("attention_mask")
                model_inputs["attention_mask"] = model_inputs.pop("decoder_attention_mask")
                model_inputs["encoder_hidden_states"] = model_inputs.pop("encoder_outputs").last_hidden_state
        print(model_kwargs_list[0].keys())
        print(models_inputs[0].keys())
        if cur_len == 2:
            troll
        outputs = [hm(**model_inputs,
                          return_dict=True,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states)
                   for hm, model_inputs in zip(hugg_models, models_inputs)]

        next_token_logits = sum(o.logits[:, -1, :] for o in outputs)

        # adjust tokens for Bart, *e.g.*
        next_token_logits = hugg_model.adjust_logits_during_generation(
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
            hm._update_model_kwargs_for_generation(o, m, is_encoder_decoder=True)
            for hm, o, m in zip(hugg_models, outputs, model_kwargs_list)]

        for m in model_kwargs_list:
            if m["past"] is not None:
                m["past"] = hugg_model._reorder_cache(m["past"], beam_idx)

        if beam_scorer.is_done:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    )

    return sequence_outputs["sequences"]


def generate(
        hugg_models,
        batch,
        encoder_outputs=None,
        num_return_sequences=None,
        max_length=None,
        num_beams=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        length_penalty=None,

):
    hugg_model = hugg_models[0]
    assert hugg_model.config.is_encoder_decoder or hugg_model.config.is_decoder
    assert "attention_mask" in batch
    assert "input_ids" in batch

    device = hugg_model.device
    do_early_stopping = hugg_model.config.early_stopping
    num_beam_groups = hugg_model.config.num_beam_groups
    length_penalty = 1.0 if length_penalty is None else length_penalty

    # The element to tile are the inputs of the decoder
    to_tile = ["last_hidden_state"] if not hasattr(hugg_model, 'to_tile') else hugg_model.to_tile
    assert isinstance(to_tile, list)

    batch = {
        argument: value for argument, value in batch.items() if not argument.startswith("decoder_")
    }

    # input_ids is now decoder_input_ids are just [bos_token_id] * bs
    input_ids = hugg_model._prepare_decoder_input_ids_for_generation(
        batch["input_ids"], bos_token_id=bos_token_id
    )

    # Huggingface seq2seq : Get each encoder output.
    if encoder_outputs is None:
        encoder_outputs = [{"encoder_outputs": hm.get_encoder()(**{**batch, **{'return_dict': True}})}
                           for hm in hugg_models]

    # get distribution pre_processing samplers
    logits_processor = hugg_model._get_logits_processor(
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        encoder_input_ids=input_ids,
        bad_words_ids=None,
        min_length=None,
        max_length=max_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=None,
        remove_invalid_values=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    )

    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    beam_scorer = BeamSearchScorer(
        batch_size=input_ids.shape[0],
        max_length=max_length,
        num_beams=num_beams,
        device=device,
        length_penalty=length_penalty,
        do_early_stopping=do_early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    # tile what is needed for decoder
    # input_ids and model_kwargs["encoder_outputs"] to beam_width
    # maybe other arguments

    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)
    for output in encoder_outputs:
        output["attention_mask"] = batch["attention_mask"].index_select(0, expanded_return_idx)
        for t in to_tile:
            to_tile_tensor = getattr(output["encoder_outputs"], t)
            setattr(output["encoder_outputs"], t,
                    to_tile_tensor.index_select(0, expanded_return_idx.to(to_tile_tensor.device)))

    return beam_search(
        hugg_model,
        hugg_models,
        input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        model_kwargs_list=encoder_outputs
    )
