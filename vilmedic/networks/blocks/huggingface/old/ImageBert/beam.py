# from . import seq2seq
# from .beam_helpers import generate
from ..beam_helpers import generate
import tqdm
import torch


def beam_search(models, config, dl):
    model = models[0]  # get model attributes
    encdecs = [model.enc_dec for model in models]
    dummy = encdecs[0]
    tgt_tokenizer = dl.dataset.tgt_tokenizer
    ref_list = []
    hyp_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            refs = batch['decoder_input_ids']
            # out = model(**batch)
            # logits = out['logits']
            # hyps = torch.argmax(logits, dim=-1)

            input_ids = batch['input_ids'].cuda()
            input_ids, _ = model.backbone(input_ids)
            input_ids = model.visual_projection(input_ids)
            hyps = generate(dummy,
                            encdecs,
                            input_ids=input_ids,
                            num_return_sequences=1,
                            max_length=dl.dataset.tgt_len,
                            num_beams=config.beam_width,
                            repetition_penalty=config.repetition_penalty,
                            # early_stopping=True,
                            bos_token_id=model.bos_token_id,
                            eos_token_id=model.eos_token_id,
                            pad_token_id=model.pad_token_id,
           )

            for h, r in zip(hyps, refs):
                hyp_list.append(tgt_tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tgt_tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return 0.0, ref_list, hyp_list
