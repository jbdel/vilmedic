import tqdm
import torch


def evaluation(models, opts, dl, **kwargs):
    hf_models = [model.dec.decoder for model in models]

    # Get tokenizer and reference sentences from dataloader
    ref_str = 'input_ids'
    tokenizer = dl.dataset.tgt_tokenizer
    max_len = dl.dataset.tgt_len

    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() for k, v in batch.items()}

            batch_size = batch['images'].shape[0]
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, opts.beam_width).view(-1).cuda()
            )

            model_kwargs = {
                "encoders_hidden_states":
                    [{"encoder_hidden_states": hf.encoder(**batch).index_select(0, expanded_return_idx)} for hf in
                     models],
                "hf_models": hf_models
            }

            hyps = hf_models[0].generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * hf_models[0].config.bos_token_id,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=opts.beam_width,
                length_penalty=opts.length_penalty,
                bos_token_id=hf_models[0].config.bos_token_id,
                eos_token_id=hf_models[0].config.eos_token_id,
                pad_token_id=hf_models[0].config.pad_token_id,
                **model_kwargs
            )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return {'refs': ref_list, 'hyps': hyp_list}
