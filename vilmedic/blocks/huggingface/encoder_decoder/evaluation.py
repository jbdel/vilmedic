import tqdm
import torch
import functools
from vilmedic.blocks.huggingface.decoder.beam_search import beam_search
from vilmedic.blocks.huggingface.decoder.beam_search import prepare_inputs_for_generation


def evaluation(models, config, dl, **kwargs):
    hf_dec = [model.dec.decoder for model in models]
    hf_enc = [model.enc for model in models]

    # We are in an ensembling scenario, we override huggingface beam-search function
    hf_dec[0].beam_search = functools.partial(beam_search, hf_dec[0])

    # Get tokenizer and reference sentences from dataloader
    ref_str = 'decoder_input_ids'
    tokenizer = dl.dataset.tgt_tokenizer
    max_len = dl.dataset.tgt_tokenizer_max_len
    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # if len(enc_dec) == 1:
            #     hyps = enc_dec[0].generate(
            #         input_ids=batch["input_ids"],
            #     )
            #
            # else:
            # Expanding inputs
            batch_size = batch['input_ids'].shape[0]
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, config.beam_width).view(-1).cuda()
            )
            # Getting encoder infos
            encoder_outputs = []
            encoder_attention_masks = []
            encoder_batch = {k: v.cuda() for k, v in batch.items() if not k.startswith("decoder_")}
            for encoder in hf_enc:
                encoder_output = encoder(**encoder_batch, return_dict=True)
                encoder_outputs.append(encoder_output.last_hidden_state)
                encoder_attention_masks.append(batch["attention_mask"])

            # Registering models and encoder outputs
            model_kwargs = {
                "encoders_outputs":
                    [
                        {"encoder_hidden_states": output.index_select(0, expanded_return_idx),
                         "encoder_attention_mask": mask.index_select(0, expanded_return_idx)
                         } for output, mask in zip(encoder_outputs, encoder_attention_masks)
                    ],
                "hf_models": hf_dec
            }

            # lets gooooo
            hyps = hf_dec[0].generate(
                inputs=torch.ones((batch_size, 1), dtype=torch.long).cuda() * hf_dec[0].config.bos_token_id,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=config.beam_width,
                length_penalty=config.length_penalty,
                bos_token_id=hf_dec[0].config.bos_token_id,
                eos_token_id=hf_dec[0].config.eos_token_id,
                pad_token_id=hf_dec[0].config.pad_token_id,
                use_cache=True,
                **model_kwargs
            )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            # break
        return {'refs': ref_list, 'hyps': hyp_list}
