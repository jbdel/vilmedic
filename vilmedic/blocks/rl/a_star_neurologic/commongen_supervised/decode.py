import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from transformers import AutoTokenizer, AutoModelWithLMHead

from commongen_supervised.generate import generate
from commongen_supervised.utils import tokenize_constraints
from commongen_supervised.lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument("--key_constraint_file", type=str, help="key elements in constraint file")

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

    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')

    with open(args.input_path) as fin:
        input_lines = [line.split('=')[0] + "=" for line in fin.read().splitlines()]

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
    key_constraints_list = read_constraints(args.key_constraint_file)

    input_lines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines]
    constraints_list = tokenize_constraints(tokenizer, constraints_list)
    key_constraints_list = tokenize_constraints(tokenizer, key_constraints_list)

    if path.exists(args.output_file):
        count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        input_lines = input_lines[count:]
        constraints_list = constraints_list[count:]
        key_constraints_list = key_constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     key_constraints=key_constraints_list[next_i:next_i + args.batch_size],
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

            outputs = generate(self=model,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               pad_token_id=PAD_ID,
                               min_length=args.min_tgt_length,
                               max_length=args.max_tgt_length,
                               num_beams=args.beam_size,
                               no_repeat_ngram_size=args.ngram_size,
                               length_penalty=args.length_penalty,
                               constraints=constraints,
                               prune_factor=args.prune_factor,
                               sat_tolerance=args.sat_tolerance,
                               look_ahead_step=args.look_ahead_step,
                               look_ahead_width=args.look_ahead_width,
                               alpha=args.alpha,
                               fusion_t=args.fusion_t,
                               look_ahead_sample=args.look_ahead_sample)

            prompt = [tokenizer.decode(x) for x in buf]
            output_sequences = [tokenizer.decode(o).split('<|endoftext|>')[0].split(prompt[i])[-1].replace('=', '').strip()
                                for i, o in enumerate(outputs)]

            for hypothesis in output_sequences:
                fout.write(hypothesis.strip().replace('<|endoftext|>', '') + "\n")
                fout.flush()

            pbar.update(1)

if __name__ == "__main__":
    main()
