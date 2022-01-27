import torch
from tqdm import tqdm


def beam_search(models, config, dl, lp_alpha=0.0):
    def tile_ctx_dict(ctx_dict, idxs):
        """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
        # 1st: tensor, 2nd optional mask
        return {
            k: (t[:, idxs], None if mask is None else mask[:, idxs])
            for k, (t, mask) in ctx_dict.items()
        }

    def check_context_ndims(ctx_dict):
        bs = []
        for name, (ctx, mask) in ctx_dict.items():
            assert ctx.dim() == 3, \
                f"{name}'s 1st dim should always be a time dimension."
            bs.append(ctx.size()[1])
        # Making sure batch_size is correct for all samples
        assert len(set(bs)) == 1
        return set(bs).pop()

    # This is the batch-size requested by the user but with sorted
    # batches, efficient batch-size will be <= max_batch_size
    max_batch_size = config.batch_size
    k = config.beam_width
    inf = -1000
    max_len = dl.dataset.tgt_len

    results = []

    # For classical models that have single encoder, decoder and
    # target vocabulary
    decs = [m.dec for m in models]
    f_inits = [dec.f_init for dec in decs]
    f_nexts = [dec.f_next for dec in decs]

    # Common parts
    encoders = [m.encode for m in models]
    # vocab = dl.dataset.tgt_vocab
    # n_vocab = len(vocab)
    # eos = vocab.word2idx(vocab.get_eos())
    # unk = vocab.word2idx(vocab.get_unk())
    # bos = vocab.word2idx(vocab.get_bos())

    #
    tgt_tokenizer = dl.dataset.tgt_tokenizer
    n_vocab = tgt_tokenizer.vocab_size
    unk = tgt_tokenizer.vocab.get(tgt_tokenizer.unk_token)
    bos = tgt_tokenizer.vocab.get(tgt_tokenizer.cls_token)
    eos = tgt_tokenizer.vocab.get(tgt_tokenizer.sep_token)

    # Tensorized beam that will shrink and grow up to max_batch_size
    beam_storage = torch.zeros(
        max_len, max_batch_size, k, dtype=torch.long).cuda()
    mask = torch.arange(max_batch_size * k).cuda()
    nll_storage = torch.zeros(max_batch_size).cuda()

    # store refs
    refs = []
    pbar = tqdm(dl, total=len(dl))
    for batch in pbar:
        # Get refs
        refs.extend(batch['decoder_input_ids'].tolist())

        # Encode source modalities
        batch = {k: v.cuda() for k, v in batch.items()}
        ctx_dicts = [encode(**batch) for encode in encoders]

        # Sanity check one of the context dictionaries for dimensions and return batch_size
        batch_size = check_context_ndims(ctx_dicts[0])

        # Always use the initial storage
        beam = beam_storage.narrow(1, 0, batch_size).zero_()

        # Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = mask.narrow(0, 0, batch_size * k)

        # nll: batch_size x 1 (will get expanded further)
        nll = nll_storage.narrow(0, 0, batch_size).unsqueeze(1)

        # Tile indices to use in the loop to expand first dim
        tile = range(batch_size)

        # Get initial decoder state (N*H)
        h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]

        # we always have <bos> tokens except that the returned embeddings
        # may differ from one model to another.

        idxs = torch.LongTensor(batch_size).fill_(bos).cuda()

        for tstep in range(max_len):
            # Select correct positions from source context
            ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

            # Get log probabilities and next state
            # log_p: batch_size x vocab_size (t = 0)
            #        batch_size*beam_size x vocab_size (t > 0)
            # NOTE: get_emb does not exist in some models, fix this.
            log_ps, h_ts = zip(
                *[f_next(cd, dec.emb(idxs), h_t[tile]) for
                  f_next, dec, cd, h_t in zip(f_nexts, decs, ctx_dicts, h_ts)])

            # Do the actual averaging of log-probabilities
            log_p = sum(log_ps).data

            # Detect <eos>'d hyps
            idxs = torch.nonzero(idxs == eos)
            if idxs.numel():
                if idxs.numel() == batch_size * k:
                    break
                idxs.squeeze_(-1)
                # Unfavor all candidates
                log_p.index_fill_(0, idxs, inf)
                # Favor <eos> so that it gets selected
                log_p.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)

            # Expand to 3D, cross-sum scores and reduce back to 2D
            # log_p: batch_size x vocab_size ( t = 0 )
            #   nll: batch_size x beam_size (x 1)
            # nll becomes: batch_size x beam_size*vocab_size here
            # Reduce (N, K*V) to k-best
            nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                batch_size, -1, n_vocab)).view(batch_size, -1).topk(
                k, sorted=False, largest=True)

            # previous indices into the beam and current token indices
            pdxs = beam[tstep] // n_vocab
            beam[tstep].remainder_(n_vocab)
            idxs = beam[tstep].view(-1)

            # Compute correct previous indices
            # Mask is needed since we're in flattened regime
            tile = pdxs.view(-1) + (nk_mask // k) * (k if tstep else 1)

            if tstep > 0:
                # Permute all hypothesis history according to new order
                beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

        # Put an explicit <eos> to make idxs_to_sent happy
        beam[max_len - 1] = eos

        # Find lengths by summing tokens not in (pad,bos,eos)
        len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

        if lp_alpha > 0.:
            len_penalty = ((5 + len_penalty) ** lp_alpha) // 6 ** lp_alpha

        # Apply length normalization
        nll.div_(len_penalty)

        # Get best-1 hypotheses
        top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
        hyps = beam[:, range(batch_size), top_hyps].t().to('cpu')
        results.extend(hyps.tolist())

    # Recover order of the samples if necessary
    if getattr(dl.batch_sampler, 'store_indices', False):
        results = [results[i] for i, j in sorted(
            enumerate(dl.batch_sampler.orig_idxs), key=lambda k: k[1])]

    hyps = [tgt_tokenizer.decode(hyp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for hyp in results]
    refs = [tgt_tokenizer.decode(ref, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ref in refs]

    return {'refs': refs, 'hyps': hyps}
