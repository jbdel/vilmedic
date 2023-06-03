import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from itertools import islice
from transformers import AutoTokenizer, AutoModelWithLMHead

from question_generation.generate import generate
from commongen_supervised.utils import tokenize_constraints
from question_generation.lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")

    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")

    parser.add_argument('--prune_factor', type=int, default=50,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    # for A star deocding
    parser.add_argument('--look_ahead_step', type=int, default=5,
                        help="number of step to look ahead")
    parser.add_argument('--look_ahead_width', type=int, default=None,
                        help="width of beam in look ahead")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")
    parser.add_argument('--fusion_t', type=float, default=None,
                        help="temperature to fuse word embedding for continuous looking ahead")
    parser.add_argument('--look_ahead_sample',  action='store_true',
                        help="whether use sampling for looking ahead")

    args = parser.parse_args()
    print(args)

    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    qmark_id = [tokenizer.convert_tokens_to_ids('?')]
    qmark_id.append(tokenizer.convert_tokens_to_ids('Ġ?'))
    eos_ids = [tokenizer.eos_token_id] + qmark_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġwho', '?"']
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

    def read_constraints(file_name):
        cons_list = []
        with open(file_name, 'r') as f:
            for line in f:
                cons = []
                for concept in json.loads(line):
                    cons.append([f' {c}' for c in concept])
                cons_list.append(cons)
        return cons_list

    constraints_list = read_constraints(args.constraint_file)

    question_words = ['What', 'When', 'Where', 'Which', 'Who', 'Whom', 'Whose', 'Why', 'How']
    beam_inits = [question_words] * len(constraints_list)
    init_factor = [len(x) for x in beam_inits]

    input_lines = [y for x in beam_inits for y in x]
    input_lines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines]

    def expand_factor(items, factors):
        expanded_items = []
        for item, factor in zip(items, factors):
            expanded_items.extend([item] * factor)
        return expanded_items

    constraints_list = tokenize_constraints(tokenizer, constraints_list)
    constraints_list = expand_factor(constraints_list, init_factor)

    if path.exists(args.output_file):
        count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        input_lines = input_lines[count:]
        constraints_list = constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    logs = []
    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     key_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            buf = _chunk
            next_i += args.batch_size

            max_len = max([len(x) for x in buf])
            buf = [x + [PAD_ID] * (max_len - len(x)) for x in buf]

            input_ids = torch.stack([torch.from_numpy(np.array(x)) for x in buf])
            input_ids = input_ids.to('cuda')
            attention_mask = (~torch.eq(input_ids, PAD_ID)).int()
            attention_mask = attention_mask.to('cuda')

            advanced_constraints = []
            for j, init_cons in enumerate(constraints):
                adv_cons = init_cons
                for token in _chunk[j // args.beam_size]:
                    adv_cons = adv_cons.advance(token)
                advanced_constraints.append(adv_cons)

            outputs, scores, sum_logprobs = generate(self=model,
                                                     input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     pad_token_id=PAD_ID,
                                                     bad_words_ids=bad_words_ids,
                                                     min_length=args.min_tgt_length,
                                                     max_length=args.max_tgt_length,
                                                     num_beams=args.beam_size,
                                                     no_repeat_ngram_size=args.ngram_size,
                                                     length_penalty=args.length_penalty,
                                                     constraints=advanced_constraints,
                                                     prune_factor=args.prune_factor,
                                                     sat_tolerance=args.sat_tolerance,
                                                     look_ahead_step=args.look_ahead_step,
                                                     look_ahead_width=args.look_ahead_width,
                                                     alpha=args.alpha,
                                                     fusion_t=args.fusion_t,
                                                     look_ahead_sample=args.look_ahead_sample)

            prompt = [tokenizer.decode(x) for x in buf]
            output_sequences = [prompt[i] + tokenizer.decode(o).split(prompt[i])[-1].split('<|endoftext|>')[0].rstrip()
                                for i, o in enumerate(outputs)]

            for hypothesis, score, sum_logprob in zip(output_sequences, scores, sum_logprobs):
                log = json.dumps({'sentence': hypothesis.strip().replace('<|endoftext|>', ''),
                                  'score': score, 'sum_logprob': sum_logprob})
                logs.append(log)
                fout.write(f'{log}\n')
                fout.flush()

            pbar.update(1)

    logs_iter = iter(logs)
    logs_list = [list(islice(logs_iter, elem)) for elem in init_factor]
    logs_list = [sorted([json.loads(s) for s in log_list], key=lambda x: x['score'], reverse=True)[0]
                 for log_list in logs_list]
    selected_outputs = [x['sentence'] for x in logs_list]

    with open(f'{args.output_file}.select', 'w') as f:
        for sentence in selected_outputs:
            f.write(f'{sentence.strip()}\n')


if __name__ == "__main__":
    main()
