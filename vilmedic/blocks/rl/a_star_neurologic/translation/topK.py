import numpy as np
import torch
import math
from torch.nn import functional as F
from scipy.stats import rankdata
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import transformers

from commongen_supervised.lexical_constraints import ConstrainedHypothesis, ConstrainedCandidate
from translation.look_ahead import _generate_beam_search, _generate_greedy, _generate_sample


# bart
def _reorder_cache(past, beam_idx):
    ((enc_out, enc_mask), decoder_cached_states) = past
    reordered_past = []
    for layer_past in decoder_cached_states:
        # get the correct batch idx from decoder layer's batch dim for cross and self-attn
        layer_past_new = {
            attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
        }
        reordered_past.append(layer_past_new)

    new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
    new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

    past = ((new_enc_out, new_enc_mask), reordered_past)
    return past

def _reorder_buffer(attn_cache, new_order):
    new_attn_cache = {}
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            new_attn_cache[k] = input_buffer_k.index_select(0, new_order)
        else:
            new_attn_cache[k] = None
    return new_attn_cache

def topk_huggingface(timestep: int,
                     batch_size: int,
                     beam_size: int,
                     vocab_size: int,
                     pad_token_id: int,
                     prune_factor: int,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     num_fill: int,
                     look_ahead_step: int,
                     look_ahead_width: int,
                     alpha: float,
                     beta: float,
                     fusion_t: float,
                     look_ahead_sample: bool,
                     init_length: int,
                     max_length: int,
                     model: transformers.PreTrainedModel,
                     temp_input_ids: torch.Tensor,
                     temp_attention_mask: torch.Tensor,
                     temp_past: Tuple[torch.Tensor],
                     decode_kwargs: Dict,
                     model_specific_kwargs
                     ) -> Tuple[np.array, np.array, List[List[Union[ConstrainedHypothesis, None]]], List[List[int]]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param vocab_size: The size of vocabulary.
    :param pad_token_id:
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: (batch_size, beam_size * target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size,))
    :param num_fill: The number of required return beam
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    seq_scores, raw_token_idx = torch.topk(scores, beam_size, dim=1, largest=True, sorted=True)
    best_ids = (raw_token_idx // vocab_size).cpu().numpy()
    best_word_ids = (raw_token_idx % vocab_size).cpu().numpy()
    seq_scores = seq_scores.cpu().numpy()

    scores = torch.reshape(scores, [batch_size, beam_size, -1]).cpu().numpy()

    select_best_ids = np.ones((batch_size, num_fill)) * -1
    select_best_word_ids = np.ones((batch_size, num_fill)) * -1
    select_seq_scores = np.zeros((batch_size, num_fill))
    select_hypotheses = [[None] * num_fill for _ in range(batch_size)]
    select_num_mets = [[-1] * num_fill for _ in range(batch_size)]

    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        idxs = torch.arange(sentno * beam_size, sentno * beam_size + beam_size, device=temp_past[0][0].device)
        if all([x is None for x in hypotheses[rows]]):
            select_best_ids[sentno] = [0] * num_fill
            select_best_word_ids[sentno] = [pad_token_id] * num_fill
            select_seq_scores[sentno] = [0] * num_fill
            select_hypotheses[sentno] = [None] * num_fill
            select_num_mets[sentno] = [-1] * num_fill
            continue

        assert not any([x is None for x in hypotheses[rows]]), 'Bad state'

        select_best_ids[sentno], select_best_word_ids[sentno], select_seq_scores[sentno],\
            select_hypotheses[sentno], select_num_mets[sentno] = _sequential_topk(sentno,
                                                                                  timestep,
                                                                                  beam_size,
                                                                                  batch_size,
                                                                                  prune_factor,
                                                                                  sat_tolerance,
                                                                                  inactive[sentno],
                                                                                  scores[sentno],
                                                                                  hypotheses[rows],
                                                                                  best_ids[sentno],
                                                                                  best_word_ids[sentno],
                                                                                  seq_scores[sentno],
                                                                                  num_fill,
                                                                                  look_ahead_step,
                                                                                  look_ahead_width,
                                                                                  alpha,
                                                                                  beta,
                                                                                  fusion_t,
                                                                                  look_ahead_sample,
                                                                                  init_length,
                                                                                  max_length,
                                                                                  model,
                                                                                  temp_input_ids[rows],
                                                                                  temp_attention_mask[rows],
                                                                                  _reorder_cache(temp_past, idxs),
                                                                                  decode_kwargs,
                                                                                  model_specific_kwargs)

    select_raw_token_idx = select_best_ids * vocab_size + select_best_word_ids
    return select_seq_scores, select_raw_token_idx, select_hypotheses, select_num_mets

# from transformers import AutoTokenizer, AutoModelWithLMHead
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

def _sequential_topk(sentno: int,
                     timestep: int,
                     beam_size: int,
                     batch_size: int,
                     prune_factor: int,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: np.array,
                     best_word_ids: np.array,
                     sequence_scores: np.array,
                     num_fill: int,
                     look_ahead_step: int,
                     look_ahead_width: int,
                     alpha: float,
                     beta: float,
                     fusion_t: float,
                     look_ahead_sample: bool,
                     init_length: int,
                     max_length: int,
                     model: transformers.PreTrainedModel,
                     temp_input_ids: torch.Tensor,
                     temp_attention_mask: torch.Tensor,
                     temp_past: Tuple[torch.Tensor],
                     decode_kwargs: Dict,
                     model_specific_kwargs,
                     chunk_size=80) -> Tuple[np.array, np.array, np.array,
                                                       List[ConstrainedHypothesis], List[int]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (beam_size,))
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    chunk_size = int(batch_size * beam_size // look_ahead_width) if chunk_size is None else chunk_size

    candidates = set()
    finished_candidates = set()
    # the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    rank = rankdata(-1 * scores, method='dense').reshape(scores.shape)

    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row, col = int(row), int(col)
        seq_score = float(seq_score)
        new_item = hypotheses[row].advance(col)
        cand = ConstrainedCandidate(row, col, seq_score, new_item)
        if hypotheses[row].finished():
            finished_candidates.add(cand)
        elif hypotheses[row].is_valid(col) or int(best_next[row]) == col:
            candidates.add(cand)

    hit = np.stack([best_ids, best_word_ids], axis=1).tolist()
    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    for row in range(beam_size if (timestep - init_length) > 0 else 1):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.positive_state.allowed()

        # (3) add the single-best item after this (if it's valid)
        best_k = np.argsort(scores[row])[::-1][:beam_size]
        for col in best_k:
            if hyp.is_valid(col):
                nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            if [row, col] not in hit and (rank[row, col] < prune_factor):
                new_item = hyp.advance(col)
                score = scores[row, col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                if hyp.finished() and col in hyp.eos():
                    finished_candidates.add(cand)
                else:
                    candidates.add(cand)

    if num_fill is not None:
        assert num_fill > beam_size, "at least select number of beam candidates"
    else:
        raise NotImplementedError

    chunk_candidates = []
    if candidates:
        # Sort the candidates.
        all_sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)

        max_satisfy = max([x.hypothesis.num_met() for x in all_sorted_candidates])
        all_sorted_candidates = [x for x in all_sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]

        if not alpha:
            future_states = [-float('inf')] * len(all_sorted_candidates)
        else:
            future_states = []
            for start in range(0, len(all_sorted_candidates), chunk_size):
                sorted_candidates = all_sorted_candidates[start: start + chunk_size]

                back_ptrs = temp_input_ids.new([x.row for x in sorted_candidates])
                curr_ids = temp_input_ids.new([x.col for x in sorted_candidates])
                input_ids = torch.cat([temp_input_ids[back_ptrs, :], curr_ids[:, None]], dim=-1)
                attention_mask = temp_attention_mask[back_ptrs, :]
                past = _reorder_cache(temp_past, back_ptrs)

                look_ahead_phrases = [c.hypothesis.phrase_to_look_ahead() for c in sorted_candidates]
                phrases_idx_map = {}
                for j, phrases in enumerate(look_ahead_phrases):
                    for phrase in phrases:
                        if tuple(phrase) not in phrases_idx_map:
                            phrases_idx_map[tuple(phrase)] = []
                        phrases_idx_map[tuple(phrase)].append(j)
                phrases_idx_mask = [(k, input_ids.new([x in v for x in range(len(sorted_candidates))])[:, None].expand(-1, look_ahead_step))
                                    for k, v in phrases_idx_map.items()]

                look_ahead_continues = [c.hypothesis.continue_to_look_ahead() for c in sorted_candidates]
                if any(look_ahead_continues):
                    continues_idx_map = {}
                    for j, continues in enumerate(look_ahead_continues):
                        for ctn in continues:
                            if tuple(ctn) not in continues_idx_map:
                                continues_idx_map[tuple(ctn)] = []
                            continues_idx_map[tuple(ctn)].append(j)
                    continues_idx_mask = [(k, torch.cat([input_ids.new([x in v for x in range(len(sorted_candidates))])[:, None],
                                                         input_ids.new_zeros(len(sorted_candidates), look_ahead_step - 1)], dim=-1))
                                          for k, v in continues_idx_map.items()]
                    phrases_idx_mask += continues_idx_mask

                look_ahead_scores = torch.full((len(phrases_idx_mask), len(sorted_candidates), look_ahead_step), -float('inf'), device=input_ids.device)

                if look_ahead_scores.shape[0]:
                    if look_ahead_sample:
                        future_state = _generate_sample(model, input_ids, look_ahead_scores, timestep + 1,
                                                        phrases_idx_mask, look_ahead_step, 0, 0.1, past,
                                                        len(sorted_candidates), look_ahead_width, attention_mask,
                                                        _reorder_cache, **decode_kwargs)
                    else:
                        if look_ahead_width == 1:
                            future_state = _generate_greedy(model, input_ids, look_ahead_scores, timestep + 1,
                                                            phrases_idx_mask, look_ahead_step, fusion_t, past,
                                                            len(sorted_candidates), attention_mask,
                                                            _reorder_cache, **decode_kwargs)
                        else:
                            future_state = _generate_beam_search(model, input_ids, look_ahead_scores, timestep + 1,
                                                                 phrases_idx_mask, look_ahead_step, past,
                                                                 len(sorted_candidates), look_ahead_width, attention_mask,
                                                                 _reorder_cache, **decode_kwargs)
                else:
                    assert all([c.hypothesis.finished()  for c in sorted_candidates])
                    future_state = [-float('inf')] * len(sorted_candidates)

                future_states.extend(future_state)

        for i, cand in enumerate(all_sorted_candidates):
            future_score = future_states[i]
            cand.rank = cand.score / (timestep - init_length + 1)
            if future_score > -10000.0 and cand.col not in cand.hypothesis.eos():
                cand.rank += alpha * future_score
            if cand.hypothesis.max_process:
                cand.rank -= beta * math.log(cand.hypothesis.max_process)
        all_sorted_candidates = sorted(all_sorted_candidates, key=attrgetter('rank'), reverse=True)

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_order() for x in all_sorted_candidates])
        grouped_candidates = [[x for x in all_sorted_candidates if x.hypothesis.met_order() == o] for o in all_orders]

        grouped_order_candidates = []
        for g in grouped_candidates:
            all_ahead = [c.ahead for c in g if c.ahead is not None]

            if not all_ahead:
                grouped_order_candidates.append(g)
                continue

            for c in g:
                c.rank = c.rank if c.ahead is not None else c.rank #+ min(all_ahead)

            grouped_order_candidates.append(sorted(g, key=attrgetter('rank'), reverse=True))

        # Group the top_i candidate of each group in chunk
        chunk_candidates = []
        num_chunk = max([len(x) for x in grouped_order_candidates])
        for i in range(num_chunk):
            chunk_i = []
            for g in grouped_order_candidates:
                if len(g) > i:
                    chunk_i.append(g[i])
            chunk_candidates.append(chunk_i)
        # Sort candidates in each chunk by score
        chunk_candidates = [sorted(x, key=attrgetter('rank'), reverse=True) for x in chunk_candidates]

    pruned_candidates = sorted(finished_candidates, key=attrgetter('score'), reverse=True)[:beam_size]
    num_finish = len(pruned_candidates)
    for chunk in chunk_candidates:
        if len(pruned_candidates) >= num_fill:
            break

        chunk = [x for x in chunk if x not in pruned_candidates]
        if not chunk:
            continue

        pruned_candidates.extend(chunk[:num_fill - len(pruned_candidates)])

    if num_fill > beam_size:
        select_num = num_finish + beam_size
        complete_candidates = sorted(pruned_candidates[:num_finish], key=attrgetter('score'), reverse=True)
        include_candidates = sorted(pruned_candidates[num_finish:select_num], key=attrgetter('score'), reverse=True)
        extra_candidates = sorted(pruned_candidates[select_num:], key=attrgetter('score'), reverse=True)
        pruned_candidates = complete_candidates + include_candidates + extra_candidates
    else:
        pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'), reverse=True)

    num_pruned_candidates = len(pruned_candidates)

    inactive = np.zeros(num_fill)
    inactive[:num_pruned_candidates] = 0

    # Pad the beam so array assignment still works
    if num_pruned_candidates < num_fill:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (num_fill - num_pruned_candidates)

    assert len(pruned_candidates) == num_fill, 'candidates number mismatch'

    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            [x.hypothesis.num_met() for x in pruned_candidates])
