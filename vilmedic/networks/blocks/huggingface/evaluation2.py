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
    if hasattr(model, "enc_dec"):
        hugg_models = [model.enc_dec for model in models]
    elif hasattr(model, "dec"):
        hugg_models = [model.dec for model in models]
    else:
        raise NotImplementedError("Model must have an enc_dec or dec attribute")

    # Get tokenizer and reference sentences from dataloader
    dataset_str = type(dl.dataset).__name__.lower()
    if 'seq2seq' in dataset_str:
        ref_str = 'decoder_input_ids'
        tokenizer = dl.dataset.tgt_tokenizer
        max_len = dl.dataset.tgt_len
    elif 'seq' in dataset_str:
        ref_str = 'input_ids'
        tokenizer = dl.dataset.tokenizer
        max_len = dl.dataset.max_len
    else:
        raise NotImplementedError("Wrong dataset for this evaluation")

    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() for k, v in batch.items()}

            # If huggingface is only the decoder, run the encoder before generate function.
            if hugg_models[0].config.is_decoder:
                encoder_outputs = [{'encoder_outputs': BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=model.encoder(**batch))} for model in models]
                # also change batch["attention_mask"] (that is supposed to decoder) to the mask of encoder (which are just ones)
                encoder_batch_size, encoder_sequence_length, _ = encoder_outputs[0][
                    "encoder_outputs"].last_hidden_state.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                encoder_attention_mask = torch.ones(encoder_hidden_shape).cuda()
                batch["attention_mask"] = encoder_attention_mask
            else:
                encoder_outputs = None

            hyps = generate(hugg_models,
                            batch=batch,
                            encoder_outputs=encoder_outputs,
                            num_return_sequences=1,
                            max_length=max_len,
                            num_beams=config.beam_width,
                            length_penalty=config.length_penalty,
                            bos_token_id=model.bos_token_id,
                            eos_token_id=model.eos_token_id,
                            pad_token_id=model.pad_token_id,
                            )
            # hyps = hugg_models[0].generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            #                                num_return_sequences=1,
            #                                max_length=max_len,
            #                                num_beams=config.beam_width,
            #                                length_penalty=config.length_penalty,
            #                                bos_token_id=model.bos_token_id,
            #                                eos_token_id=model.eos_token_id,
            #                                pad_token_id=model.pad_token_id,
            #                                )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return {'refs': ref_list, 'hyps': hyp_list}
