# from . import seq2seq
from .beam_helpers import generate
import tqdm
import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

"""
Performs beam-search and ensembling.
Use this function for huggingface-based model
Either you are using a seq2seq pipeline, in which case your seq2seq attribute should be called "enc_dec".
Or you are using only a decoder, in which case your dec attribute should be called "dec"
"""


def beam_search(models, config, dl):
    # Get huggingface object responsible of generation (encoder_decoder model or decoder model)
    model = models[0]
    hugg_models = [model.dec for model in models]

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
                "encoder_hidden_states": model.encode(**batch).index_select(0, expanded_return_idx)
            }

            hyps = hugg_models[0].generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * model.bos_token_id,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=config.beam_width,
                length_penalty=config.length_penalty,
                bos_token_id=model.bos_token_id,
                eos_token_id=model.eos_token_id,
                pad_token_id=model.pad_token_id,
                **model_kwargs
            )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return {'refs': ref_list, 'hyps': hyp_list}
