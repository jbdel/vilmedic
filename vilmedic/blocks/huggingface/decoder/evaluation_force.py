import tqdm
import torch
import functools
from .beam_search import beam_search
import torch.nn as nn


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

    # We are in an ensembling scenario, we override huggingface beam-search function
    hf_models[0].beam_search = functools.partial(beam_search, hf_models[0])

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
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Expanding inputs
            batch_size = batch[ref_str].shape[0]
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, config.beam_width).view(-1).cuda()
            )

            print([[phrase.split() for phrase in sublist] for sublist in batch["concepts"]])
            force_words_ids = tokenizer([[phrase.split() for phrase in sublist] for sublist in batch["concepts"]],
                                        add_special_tokens=False)
            print(force_words_ids)
            troll

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

            # lets gooooo
            hyps = hf_models[0].generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * bos_token_id,
                num_return_sequences=1,
                max_length=max_len,
                force_words_ids=force_words_ids,
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
