import sys
from vilmedic.blocks.scorers.scores import compute_scores
import logging
import json


def main():
    try:
        hyps_file = sys.argv[1]
        refs_file = sys.argv[2]
        metrics = sys.argv[3].split(",")
        print("Hyp file is", hyps_file)
        print("Ref file is", refs_file)
        print("Metrics are", metrics)
    except IndexError:
        print("Usage vilmedic-metrics hyps_file refs_file metric1,metric2,...")
        return

    try:
        hyps = [line.strip() for line in open(hyps_file).readlines()]
        refs = [line.strip() for line in open(refs_file).readlines()]
    except FileNotFoundError as e:
        print(e)
        return

    print("Computing metrics, this can take a while...")
    print(json.dumps(compute_scores(metrics,
                                    refs=refs,
                                    hyps=hyps,
                                    split=None,
                                    seed=None,
                                    config=None,
                                    epoch=None,
                                    logger=logging.getLogger(__name__),
                                    dump=False),
                     indent=4)
          )


if __name__ == '__main__':
    main()
