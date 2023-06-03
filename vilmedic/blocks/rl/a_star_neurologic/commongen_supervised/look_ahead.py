import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def _expand_input(batch_size, num_beams, input_ids, attention_mask, position_ids,
                  look_ahead_scores, phrases_idx_mask, past, _reorder_cache):
    input_ids_len = input_ids.shape[-1]
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, input_ids_len)
    input_ids = input_ids.contiguous().view(batch_size * num_beams, input_ids_len)

    position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, input_ids_len)
    position_ids = position_ids.contiguous().view(batch_size * num_beams, input_ids_len)

    attention_mask_len = attention_mask.shape[-1]
    attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, attention_mask_len)
    attention_mask = attention_mask.contiguous().view(batch_size * num_beams, attention_mask_len)

    beam_idx = input_ids.new([i for i in range(batch_size) for _ in range(num_beams)])
    past = _reorder_cache(past, beam_idx)

    look_ahead_len = look_ahead_scores.shape[0]
    look_ahead_scores = look_ahead_scores.unsqueeze(2).expand(look_ahead_len, batch_size, num_beams, -1)
    look_ahead_scores = look_ahead_scores.contiguous().view(look_ahead_len, batch_size * num_beams, -1)

    extended_phrases_idx_mask = []
    for phrase, mask in phrases_idx_mask:
        mask = mask.unsqueeze(1).expand(batch_size, num_beams, -1)
        mask = mask.contiguous().view(batch_size * num_beams, -1)
        extended_phrases_idx_mask.append((phrase, mask))

    return input_ids, attention_mask, position_ids, look_ahead_scores, extended_phrases_idx_mask, past


def _generate_greedy(
        self,
        input_ids,
        look_ahead_scores,
        cur_len,
        phrases_idx_mask,
        look_ahead_step,
        fusion_t,
        past,
        batch_size,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(-1)
    sent_scores = input_ids.new(batch_size).fill_(0).float()
    word_embeds = None

    for t in range(look_ahead_step):
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1)
        if word_embeds is not None:
            del model_inputs["input_ids"]
            model_inputs["inputs_embeds"] = word_embeds.unsqueeze(1)

        outputs = self(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        log_prob = F.log_softmax(next_token_logits, dim=-1)

        scores = postprocess_next_token_scores(
            self=self,
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
            pad_token_id=pad_token_id,
        )

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]

        # Greedy decoding
        next_token = torch.argmax(scores, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        scores_to_add = torch.gather(log_prob, 1, tokens_to_add[:, None]).squeeze(1)
        scores_to_add *= (tokens_to_add != pad_token_id).float()
        sent_scores += scores_to_add

        # compute the score for phrases to appear in the future
        for j, (phrase, phrase_mask) in enumerate(phrases_idx_mask):
            if torch.sum(phrase_mask[:, t]).item():
                phrase_score = log_prob[:, phrase[0]]

                if len(phrase) > 1:
                    phrase_input_ids = input_ids.new(phrase)[None, :].expand(batch_size, -1)
                    phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(phrase))], dim=-1)
                    phrase_attention_mask = torch.cat(
                        [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
                                            for _ in range(len(phrase))], dim=-1
                    )

                    follow_logits = self(input_ids=phrase_input_ids, past=past,
                                         attention_mask=phrase_attention_mask, position_ids=phrase_position_ids,
                                         labels=phrase_input_ids, use_cache=use_cache)[1]
                    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

                    phrase_score = phrase_score[:, None]
                    for i in range(len(phrase[:-1])):
                        phrase_score = torch.cat([phrase_score, follow_log_prob[:, i, phrase[i + 1]][:, None]], dim=-1)
                    phrase_score = torch.mean(phrase_score, dim=-1)

                phrase_score.masked_fill_(phrase_mask[:, t] == 0, -float("inf"))
                phrase_score.masked_fill_(unfinished_sents == 0, -float("inf"))
                look_ahead_scores[j, :, t] = phrase_score

        # update word embedding for current tokens_to_add
        if fusion_t is not None:
            scores = scores / fusion_t
            probs = F.softmax(scores, dim=-1)
            word_embeds = torch.matmul(probs, self.get_input_embeddings().weight)

            if eos_token_id is not None:
                pad_one_hot = F.one_hot(next_token.new(next_token.shape).fill_(pad_token_id), num_classes=vocab_size)
                pad_embed = torch.matmul(pad_one_hot.float(), self.get_input_embeddings().weight)
                word_embeds = word_embeds * unfinished_sents[:, None] + pad_embed * (1 - unfinished_sents[:, None])

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # return states
    look_ahead_scores = torch.max(torch.max(look_ahead_scores, dim=2)[0], dim=0)[0]
    return look_ahead_scores.tolist()


def _generate_sample(
        self,
        input_ids,
        look_ahead_scores,
        cur_len,
        phrases_idx_mask,
        look_ahead_step,
        look_ahead_top_k,
        look_ahead_top_p,
        past,
        batch_size,
        num_samples,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """

    input_ids, attention_mask, position_ids, look_ahead_scores, phrases_idx_mask, past = \
        _expand_input(batch_size, num_samples, input_ids, attention_mask, position_ids,
                      look_ahead_scores, phrases_idx_mask, past, _reorder_cache)

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size * num_samples).fill_(1)
    sent_lengths = input_ids.new(batch_size * num_samples).fill_(-1)
    sent_scores = input_ids.new(batch_size * num_samples).fill_(0).float()

    for t in range(look_ahead_step):
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1)

        outputs = self(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        log_prob = F.log_softmax(next_token_logits, dim=-1)

        scores = postprocess_next_token_scores(
            self=self,
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
            pad_token_id=pad_token_id,
        )

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]

        # Top-p/top-k filtering
        next_token_logscores = top_k_top_p_filtering(scores.clone(), top_k=look_ahead_top_k, top_p=look_ahead_top_p)
        # Sample
        probs = F.softmax(next_token_logscores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        scores_to_add = torch.gather(log_prob, 1, tokens_to_add[:, None]).squeeze(1)
        scores_to_add *= (tokens_to_add != pad_token_id).float()
        sent_scores += scores_to_add

        # compute the score for phrases to appear in the future
        for j, (phrase, phrase_mask) in enumerate(phrases_idx_mask):
            if torch.sum(phrase_mask[:, t]).item():
                phrase_score = log_prob[:, phrase[0]]

                if len(phrase) > 1:
                    phrase_input_ids = input_ids.new(phrase)[None, :].expand(input_ids.shape[0], -1)
                    phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(phrase))], dim=-1)
                    phrase_attention_mask = torch.cat(
                        [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
                                            for _ in range(len(phrase))], dim=-1
                    )

                    follow_logits = self(input_ids=phrase_input_ids, past=past,
                                         attention_mask=phrase_attention_mask, position_ids=phrase_position_ids,
                                         labels=phrase_input_ids, use_cache=use_cache)[1]
                    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

                    phrase_score = phrase_score[:, None]
                    for i in range(len(phrase[:-1])):
                        phrase_score = torch.cat([phrase_score, follow_log_prob[:, i, phrase[i + 1]][:, None]], dim=-1)
                    phrase_score = torch.mean(phrase_score, dim=-1)

                phrase_score.masked_fill_(phrase_mask[:, t] == 0, -float("inf"))
                phrase_score.masked_fill_(unfinished_sents == 0, -float("inf"))
                look_ahead_scores[j, :, t] = phrase_score

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # return states
    look_ahead_scores = torch.max(torch.max(look_ahead_scores, dim=2)[0], dim=0)[0]
    look_ahead_scores = torch.max(torch.reshape(look_ahead_scores, (batch_size, num_samples)), dim=1)[0]
    return look_ahead_scores.tolist()


def _generate_beam_search(
        self,
        input_ids,
        look_ahead_scores,
        cur_len,
        phrases_idx_mask,
        look_ahead_step,
        past,
        batch_size,
        num_beams,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """

    input_ids, attention_mask, position_ids, look_ahead_scores, phrases_idx_mask, past = \
        _expand_input(batch_size, num_beams, input_ids, attention_mask, position_ids,
                      look_ahead_scores, phrases_idx_mask, past, _reorder_cache)

    # generated hypotheses
    generated_hyps = [
        FinishHypotheses(num_beams) for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # done sentences
    done = [False for _ in range(batch_size)]

    for t in range(look_ahead_step):
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1)

        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]
        if self.config.is_encoder_decoder and do_sample is False:
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        # compute the score for phrases to appear in the future
        for j, (phrase, phrase_mask) in enumerate(phrases_idx_mask):
            if torch.sum(phrase_mask[:, t]).item():
                phrase_score = scores[:, phrase[0]]

                if len(phrase) > 1:
                    phrase_input_ids = input_ids.new(phrase)[None, :].expand(input_ids.shape[0], -1)
                    phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(phrase))], dim=-1)
                    phrase_attention_mask = torch.cat(
                        [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
                                            for _ in range(len(phrase))], dim=-1
                    )

                    follow_logits = self(input_ids=phrase_input_ids, past=past,
                                         attention_mask=phrase_attention_mask, position_ids=phrase_position_ids,
                                         labels=phrase_input_ids, use_cache=use_cache)[1]
                    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

                    phrase_score = phrase_score[:, None]
                    for i in range(len(phrase[:-1])):
                        phrase_score = torch.cat([phrase_score, follow_log_prob[:, i, phrase[i + 1]][:, None]], dim=-1)
                    phrase_score = torch.mean(phrase_score, dim=-1)

                phrase_score.masked_fill_(phrase_mask[:, t] == 0, -float("inf"))
                is_finish = torch.reshape(torch.tensor(done, device=input_ids.device)[:, None].expand(-1, num_beams),
                                          (batch_size * num_beams,)).int()
                phrase_score.masked_fill_(is_finish == 1, -float("inf"))
                look_ahead_scores[j, :, t] = phrase_score

        scores = postprocess_next_token_scores(
            self=self,
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        position_ids = position_ids[beam_idx, :]
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        cur_len = cur_len + 1

        # re-order internal states
        if past is not None:
            past = self._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # return states
    look_ahead_scores = torch.max(torch.max(look_ahead_scores, dim=2)[0], dim=0)[0]
    look_ahead_scores = torch.max(torch.reshape(look_ahead_scores, (batch_size, num_beams)), dim=1)[0]
    return look_ahead_scores.tolist()


class FinishHypotheses(object):
    def __init__(self, num_beams):
        """
        Initialize n-best list of hypotheses.
        """
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        if len(self) < self.num_beams or sum_logprobs > self.worst_score:
            self.beams.append((sum_logprobs, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(sum_logprobs, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        else:
            ret = self.worst_score >= best_sum_logprobs
            return ret


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
        pad_token_id,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        self.enforce_repetition_penalty_(
            scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None:
        current_len = torch.sum(input_ids != pad_token_id, dim=1).int()
        for idx in range(batch_size * num_beams):
            if current_len[idx] < min_length:
                scores[idx, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, pad_token_id
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, pad_token_id: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    current_len = torch.sum(prev_input_ids != pad_token_id, dim=1).int()
    if max(current_len).item() + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = [x for x in prev_input_ids[idx].tolist() if x != pad_token_id]
        if len(gen_tokens) < no_repeat_ngram_size:
            continue
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        gen_tokens = [x for x in prev_input_ids[hypo_idx].tolist() if x != pad_token_id]
        start_idx = len(gen_tokens) + 1 - no_repeat_ngram_size
        ngram_idx = tuple(gen_tokens[start_idx:len(gen_tokens)])
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens