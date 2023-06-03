import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True,
                        help='raw generated file to evaluate')
    parser.add_argument("--constraint_file", type=str,
                        default='machine_translation/iate/iate.414.terminology.tsv')

    args = parser.parse_args()

    constraint_lines = [l.strip() for l in open(args.constraint_file, 'r').readlines()]
    constraints = []
    for line in constraint_lines:
        words = line.split('\t')[2:]
        words = [x for i, x in enumerate(words) if i % 2 == 1]
        constraints.append(words)

    percents = []
    input_lines = [l for l in open(args.input_file, 'r').readlines()]
    for words, sent in zip(constraints, input_lines):
        percent = sum([int(w in sent) for w in words]) / len(words)
        percents.append(percent)

    print(f'Average coverage: {sum(percents) / len(percents)}')
