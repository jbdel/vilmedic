import tqdm
import torch
import functools
from .beam_search import beam_search


def evaluation(models, config, dl, **kwargs):
    hf_models = [model.dec.decoder for model in models]

    # We are in an ensembling scenario, we override huggingface beam-search function
    hf_models[0].beam_search = functools.partial(beam_search, hf_models[0])

    # Get tokenizer and reference sentences from dataloader
    ref_str = 'input_ids'
    tokenizer = dl.dataset.tokenizer
    max_len = dl.dataset.tokenizer_max_len
    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            # Expanding inputs
            batch_size = batch['images'].shape[0]
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

            # lets gooooo
            hyps = hf_models[0].generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * hf_models[0].config.bos_token_id,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=config.beam_width,
                length_penalty=config.length_penalty,
                bos_token_id=hf_models[0].config.bos_token_id,
                eos_token_id=hf_models[0].config.eos_token_id,
                pad_token_id=hf_models[0].config.pad_token_id,
                use_cache=True,
                **model_kwargs
            )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            # break
        return {'refs': ref_list, 'hyps': hyp_list}
