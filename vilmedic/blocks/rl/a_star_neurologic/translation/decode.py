import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from transformers import MarianTokenizer, MarianMTModel

from translation.generate import generate
from translation import utils_seq2seq
from commongen_supervised.lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
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
    parser.add_argument('--alpha', type=float, default=0.,
                        help="decay factor for score in looking ahead")
    parser.add_argument('--beta', type=float, default=0.,
                        help="reward factor for in progress constraint")
    parser.add_argument('--fusion_t', type=float, default=None,
                        help="temperature to fuse word embedding for continuous looking ahead")
    parser.add_argument('--look_ahead_sample',  action='store_true',
                        help="whether use sampling for looking ahead")

    args = parser.parse_args()
    print(args)

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    eos_ids = [tokenizer.eos_token_id]

    input_lines = [l.strip() for l in open(args.input_path, 'r').readlines()]

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
    constraints_list = utils_seq2seq.tokenize_constraints(tokenizer, constraints_list)

    if path.exists(args.output_file):
        count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        input_lines = input_lines[count:]
        constraints_list = constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0
    scores_list, sum_logprobs_list, ppl_list = [], [], []

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            src_text = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     key_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            next_i += args.batch_size

            inputs = tokenizer(src_text, return_tensors='pt', padding=True)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')

            outputs, scores, sum_logprobs = generate(self=model,
                                                     input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     min_length=args.min_tgt_length,
                                                     max_length=args.max_tgt_length,
                                                     no_repeat_ngram_size=args.ngram_size,
                                                     num_beams=args.beam_size,
                                                     length_penalty=args.length_penalty,
                                                     constraints=constraints,
                                                     prune_factor=args.prune_factor,
                                                     sat_tolerance=args.sat_tolerance,
                                                     look_ahead_step=args.look_ahead_step,
                                                     look_ahead_width=args.look_ahead_width,
                                                     alpha=args.alpha,
                                                     beta=args.beta,
                                                     fusion_t=args.fusion_t,
                                                     look_ahead_sample=args.look_ahead_sample
                                                     )
            output_sequences = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            scores_list.extend(scores)
            sum_logprobs_list.extend(sum_logprobs)

            output_length = [len([y for y in x.tolist() if y != model.config.pad_token_id]) for x in outputs]
            ppl = [-s / l for s, l in zip(sum_logprobs, output_length)]
            ppl_list.extend(ppl)

            for hypothesis in output_sequences:
                fout.write(hypothesis.strip() + "\n")
                fout.flush()

            pbar.update(1)

    print(args.output_file)
    print(f'Average score: {sum(scores_list) / len(scores_list)}')
    print(f'Average sum logprob: {sum(sum_logprobs_list) / len(sum_logprobs_list)}')
    print(f'Average PPL: {sum(ppl_list) / len(ppl_list)}')

if __name__ == "__main__":
    main()
