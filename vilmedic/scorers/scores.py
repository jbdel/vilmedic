import os
from .cocoeval import Rouge
import shutil
import pathlib
import subprocess
from six.moves import zip_longest
from rouge_score import rouge_scorer
from rouge_score import scoring
import numpy as np


def meteor_score(refs, hyps, language="en"):
    jar = "./vilmedic/scorers/cocoeval/meteor/meteor-1.5.jar"
    __cmdline = ["java", "-Xmx2G", "-jar", jar,
                 "-", "-", "-stdio"]
    assert os.path.exists(jar), 'meteor jar not found'
    env = os.environ
    env['LC_ALL'] = 'en_US.UTF-8'

    # Sanity check
    if shutil.which('java') is None:
        raise RuntimeError('METEOR requires java which is not installed.')

    cmdline = __cmdline[:]
    refs = [refs] if not isinstance(refs, list) else refs

    if isinstance(hyps, str):
        # If file, open it for line reading
        hyps = open(hyps)

    if language == "auto":
        # Take the extension of the 1st reference file, e.g. ".de"
        language = pathlib.Path(refs[0]).suffix[1:]

    cmdline.extend(["-l", language])

    # Make reference files a list
    iters = [open(f) for f in refs]
    iters.append(hyps)

    # Run METEOR process
    proc = subprocess.Popen(cmdline,
                            stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env,
                            universal_newlines=True, bufsize=1)

    eval_line = 'EVAL'

    for line_ctr, lines in enumerate(zip(*iters)):
        lines = [l.rstrip('\n') for l in lines]
        refstr = " ||| ".join(lines[:-1])
        line = "SCORE ||| " + refstr + " ||| " + lines[-1]

        proc.stdin.write(line + '\n')
        eval_line += ' ||| {}'.format(proc.stdout.readline().strip())

    # Send EVAL line to METEOR
    proc.stdin.write(eval_line + '\n')

    # Dummy read segment scores
    for i in range(line_ctr + 1):
        proc.stdout.readline().strip()

    # Compute final METEOR
    try:
        score = float(proc.stdout.readline().strip())
    except Exception as e:
        score = 0.0
    finally:
        # Close METEOR process
        proc.stdin.close()
        proc.terminate()
        proc.kill()
        proc.wait(timeout=2)
        return score


def rouge_score(refs, hyps):
    assert len(hyps) == len(refs), "ROUGE: # of sentences does not match."

    rouge_scorer = Rouge()

    rouge_sum = 0
    for hyp, ref in zip(hyps, refs):
        rouge_sum += rouge_scorer.calc_score([hyp], [ref])

    score = (rouge_sum) / len(hyps)

    return score


def bleu_score(refs, hyps):
    # -*- coding: utf-8 -*-
    import subprocess
    import pkg_resources
    BLEU_SCRIPT = './vilmedic/scorers/multi-bleu.perl'
    cmdline = [BLEU_SCRIPT]

    # Make reference files a list
    refs = [refs] if not isinstance(refs, list) else refs
    cmdline.extend(refs)

    if isinstance(hyps, str):
        hypstring = open(hyps).read().strip()
    elif isinstance(hyps, list):
        hypstring = "\n".join(hyps)

    score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                           input=hypstring,
                           universal_newlines=True).stdout.splitlines()

    if len(score) == 0:
        return 0.0
    else:
        score = score[0].strip()
        float_score = float(score.split()[2][:-1])
        verbose_score = score.replace('BLEU = ', '')
        return float_score


def google_rouge(refs, hyps, rouges):
    scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
    scores = []
    for target_rec, prediction_rec in zip_longest(refs, hyps):
        if target_rec is None or prediction_rec is None:
            raise ValueError("Must have equal number of lines across target and "
                             "prediction.")
        scores.append(scorer.score(target_rec, prediction_rec))

    # aggregator = scoring.BootstrapAggregator()
    # for score in scores:
    #     aggregator.add_scores(score)
    # print(aggregator.aggregate())
    return np.mean([s[rouges[0]].fmeasure for s in scores])


def accuracy(refs, hyps):
    return np.mean(np.array(refs) == np.array(hyps))


def compute_scores(metrics, refs, hyps, split, seed, ckpt_dir, epoch):
    assert len(refs) == len(hyps), '{} vs {}'.format(len(refs), len(hyps))

    # Dump
    base = os.path.join(ckpt_dir, '{}_{}_{}'.format(split, seed, '{}'))
    refs_file = base.format('refs.txt')
    hyps_file = base.format('hyps.txt')
    metrics_file = base.format('metrics.txt')

    with open(refs_file, 'w') as f:
        f.write('\n'.join(map(str, refs)))

    with open(hyps_file, 'w') as f:
        f.write('\n'.join(map(str, hyps)))

    scores = {}
    for metric in metrics:
        if metric == 'BLEU':
            scores["BLEU"] = round(bleu_score(refs_file, hyps_file), 2)
        elif metric == 'ROUGE1':
            scores["ROUGE1"] = round(google_rouge(refs, hyps, rouges=['rouge1']) * 100, 2)
        elif metric == 'ROUGE2':
            scores["ROUGE2"] = round(google_rouge(refs, hyps, rouges=['rouge2']) * 100, 2)
        elif metric == 'ROUGEL':
            scores["ROUGEL"] = round(google_rouge(refs, hyps, rouges=['rougeL']) * 100, 2)
        elif metric == 'METEOR':
            scores["METEOR"] = round(meteor_score(refs_file, hyps_file) * 100, 2)
        elif metric == 'accuracy':
            scores["accuracy"] = round(accuracy(refs, hyps) * 100, 2)
        else:
            raise NotImplementedError(metric)

    with open(metrics_file, 'a+') as f:
        f.write(str({
            'split': split,
            'epoch': epoch,
            'scores': scores,
        }) + '\n')

    return scores
