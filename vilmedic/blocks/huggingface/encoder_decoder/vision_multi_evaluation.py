import tqdm
import torch
from transformers import VisionEncoderDecoderModel, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput

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
    # Handle both regular models and DataParallel-wrapped models
    first_model = models[0]
    if hasattr(first_model, 'model'):
        hf_model: VisionEncoderDecoderModel = first_model.model
    elif hasattr(first_model, 'module'):
        # DataParallel case - the actual model is in .module
        hf_model: VisionEncoderDecoderModel = first_model.module.model
    else:
        # Direct model case
        hf_model: VisionEncoderDecoderModel = first_model

    # pick tokenizer / ref field
    try:
        ref_str = 'input_ids'
        tokenizer = dl.dataset.tokenizer
        max_len = dl.dataset.tokenizer_max_len
    except AttributeError:
        ref_str = 'decoder_input_ids'
        tokenizer = dl.dataset.tgt_tokenizer
        max_len = dl.dataset.tgt_tokenizer_max_len

    refs, hyps = [], []
    bos_id, eos_id, pad_id = get_special_token_ids(hf_model, tokenizer)

    # base generation args
    gen_args = {
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "num_return_sequences": 1,
        "max_length": max_len,
        "use_cache": True,
    }
    if getattr(config, "length_penalty", None) is not None:
        gen_args["length_penalty"] = config.length_penalty
    if getattr(config, "beam_width", None) is not None:
        gen_args["num_beams"] = config.beam_width   # fixed typo here

    gen_conf = GenerationConfig(**gen_args, decoder_start_token_id=tokenizer.cls_token_id)

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            # to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            images   = batch["images"]                   # [B, C, H, W] or [B, N, C, H, W]
            img_mask = batch.get("images_mask", None)    # None or [B, N]

            # --- MULTI-IMAGE ---
            if images.dim() == 5:
                B, N, C, H, W = images.shape

                # build or default the image‐presence mask
                if img_mask is None:
                    img_mask = torch.ones((B, N), dtype=torch.bool, device=images.device)
                else:
                    img_mask = img_mask.to(images.device).bool()

                # flatten & encode
                flat = images.view(B * N, C, H, W)
                enc_out = hf_model.encoder(
                    pixel_values=flat,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                flat_hidden = enc_out.last_hidden_state        # [B*N, S, D]
                S, D = flat_hidden.size(1), flat_hidden.size(2)

                # concat along sequence dim → [B, N*S, D]
                concat_hidden = flat_hidden.view(B, N * S, D)

                # project if needed
                if (hf_model.encoder.config.hidden_size != hf_model.decoder.config.hidden_size
                    and hf_model.decoder.config.cross_attention_hidden_size is None):
                    concat_hidden = hf_model.enc_to_dec_proj(concat_hidden)

                # expand image mask to patch mask → [B, N*S]
                attn_mask = img_mask.unsqueeze(-1).expand(B, N, S).reshape(B, N * S).long()

                # generate with precomputed states + mask                
                encoder_outputs = BaseModelOutput(last_hidden_state=concat_hidden)

                out_ids = hf_model.generate(
                    generation_config=gen_conf,          # your GenerationConfig
                    encoder_outputs=encoder_outputs,     # goes in kwargs
                    attention_mask=attn_mask,            # encoder attention mask
                )

            # --- SINGLE-IMAGE ---
            elif images.dim() == 4:
                out_ids = hf_model.generate(
                    images,
                    generation_config=gen_conf
                )

            else:
                raise NotImplementedError(f"Unexpected images.dim() = {images.dim()}")

            # decode & collect
            for pred_ids, ref_ids in zip(out_ids, batch[ref_str]):
                hyps.append(
                    tokenizer.decode(pred_ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
                )
                refs.append(
                    tokenizer.decode(ref_ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
                )

    return {"refs": refs, "hyps": hyps}
