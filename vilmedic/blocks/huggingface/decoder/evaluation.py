import copy
from weakref import ref
import tqdm
import torch
import functools
from .beam_search import beam_search, constrained_beam_search
import torch.nn as nn

from vilmedic.blocks.rl.a_star_neurologic_utils import a_star_generate, init_batch, process_constraints

def get_special_token_ids(model, tokenizer):
    bos_token_id = model.config.bos_token_id
    eos_token_id = model.config.eos_token_id
    pad_token_id = model.config.pad_token_id
    if None in [bos_token_id, eos_token_id, pad_token_id]:
        bos_token_id = tokenizer.vocab[tokenizer.cls_token]
        eos_token_id = tokenizer.vocab[tokenizer.sep_token]
        pad_token_id = tokenizer.vocab[tokenizer.pad_token]

    return bos_token_id, eos_token_id, pad_token_id


def evaluation(models, config, dl, **kwargs):
    models = [m if not isinstance(m, nn.DataParallel) else m.module for m in models]
    hf_models = [model.dec.decoder for model in models]

    # We are in an ensembling scenario, we override huggingface beam-search functions
    hf_models[0].beam_search = functools.partial(beam_search, hf_models[0])
    # hf_models[0].constrained_beam_search = functools.partial(constrained_beam_search, hf_models[0])

    # Get tokenizer and reference sentences from dataloader
    try:
        ref_str = 'input_ids'
        tokenizer = dl.dataset.tokenizer
        max_len = dl.dataset.tokenizer_max_len
    except AttributeError:
        ref_str = 'decoder_input_ids'
        tokenizer = dl.dataset.tgt_tokenizer
        max_len = dl.dataset.tgt_tokenizer_max_len

    # Get tokens
    bos_token_id, eos_token_id, pad_token_id = get_special_token_ids(hf_models[0], tokenizer)

    ref_list = []
    hyp_list = []

    with torch.no_grad():
        # next_i = 0
        for batch in tqdm.tqdm(dl):
            force_input_ids = batch.pop('force_input_ids', [])
            batch = {k: v.cuda() for k, v in batch.items()}

            # Expanding inputs
            batch_size = batch[ref_str].shape[0]

            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, config.beam_width).view(-1).cuda()
            )
            # Getting encoder infos
            encoder_outputs = []
            encoder_attention_masks = []
            for hf in models:
                encoder_output, encoder_attention_mask = hf.encode(**batch)
                encoder_outputs.append(encoder_output)
                encoder_attention_masks.append(encoder_attention_mask)

            # Registering models and encoder outputs
            model_kwargs = {
                "encoders_outputs":
                    [
                        {"encoder_hidden_states": output.index_select(0, expanded_return_idx),
                         "encoder_attention_mask": mask.index_select(0, expanded_return_idx)
                         } for output, mask in zip(encoder_outputs, encoder_attention_masks)
                    ],
                "hf_models": hf_models
            }
            # let's gooooo
            if force_input_ids and len(force_input_ids[0]) > 0 and len(force_input_ids[0][0]) > 0:
                assert batch_size == len(force_input_ids) == 1, "To use constraint forcing, batch_size must be 1"
                input_ids = torch.ones((batch_size, 1), dtype=torch.long).cuda() * bos_token_id
                # decoded_constraints = [tokenizer.decode(force_input_id[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False) for force_input_id in force_input_ids]
                # print(decoded_constraints)

                # constraints_list = process_constraints(force_input_ids)

                # constraints = init_batch(
                #     # raw_constraints=constraints_list[next_i:next_i + batch_size],
                #     # key_constraints=constraints_list[next_i:next_i + batch_size],
                #     raw_constraints=constraints_list,
                #     key_constraints=constraints_list,
                #     beam_size=config.beam_width,
                #     eos_id=[eos_token_id] + [tokenizer('.')]
                # )

                # next_i += batch_size
                
                # TODO: min_length, no_repeat_ngram_size, length_penalty? unclear if output_scores and return_dict_in_generate are handled in `a_star_generate`
                # hyps, _, _ = a_star_generate(
                #     self=hf_models[0],
                #     input_ids=input_ids,
                #     attention_mask=(~torch.eq(input_ids, bos_token_id)).int(),
                #     pad_token_id=pad_token_id,
                #     max_length=max_len,
                #     num_beams=config.beam_width,
                #     num_return_sequences=1,
                #     bad_words_ids=[[pad_token_id], [bos_token_id]],
                #     constraints=constraints,
                #     prune_factor=config.prune_factor,
                #     sat_tolerance=config.sat_tolerance,
                #     alpha=config.alpha,
                #     beta=config.beta,
                #     look_ahead_step=config.look_ahead_step,
                #     look_ahead_width=config.look_ahead_width,
                #     fusion_t=config.fusion_t,
                #     look_ahead_sample=config.look_ahead_sample,
                #     do_sample=False,
                #     use_cache=True,
                #     **model_kwargs
                # )

                model_kwargs = {
                    "encoder_hidden_states": model_kwargs['encoders_outputs'][0]['encoder_hidden_states'],
                    "encoder_attention_mask": model_kwargs['encoders_outputs'][0]['encoder_attention_mask'],
                }

                hyps = hf_models[0].generate(
                    input_ids,
                    force_words_ids=force_input_ids[0],
                    max_length=max_len,
                    num_beams=config.beam_width,
                    length_penalty=config.length_penalty,
                     bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    remove_invalid_values=True,
                    use_cache=True,
                    **model_kwargs
                )

                # decoded_hyps = tokenizer.decode(hyps[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print(decoded_hyps)
            else:
                hyps = hf_models[0].generate(
                    input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * bos_token_id,
                    num_return_sequences=1,
                    max_length=max_len,
                    num_beams=config.beam_width,
                    length_penalty=config.length_penalty,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                    **model_kwargs
                )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            # break
        return {'refs': ref_list, 'hyps': hyp_list}
