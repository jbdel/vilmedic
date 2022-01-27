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
            batch_size = batch['images'].shape[0]
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, config.beam_width).view(-1).cuda()
            )

            model_kwargs = {
                "encoders_hidden_states":
                    [{"encoder_hidden_states": hf.encode(**batch).index_select(0, expanded_return_idx)} for hf in
                     models],
                "hf_models": hf_models
            }

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

        return {'refs': ref_list, 'hyps': hyp_list}
