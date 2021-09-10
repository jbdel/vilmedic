import subprocess


class BLEUScorer:
    def __init__(self):

        BLEU_SCRIPT = './vilmedic/scorers/nlg/bleu/multi-bleu.perl'
        self.cmdline = [BLEU_SCRIPT]

    def compute(self, refs, hyps):
        refs = [refs] if not isinstance(refs, list) else refs
        self.cmdline.extend(refs)

        if isinstance(hyps, str):
            hypstring = open(hyps).read().strip()
        elif isinstance(hyps, list):
            hypstring = "\n".join(hyps)

        score = subprocess.run(self.cmdline,
                               stdout=subprocess.PIPE,
                               input=hypstring,
                               universal_newlines=True).stdout.splitlines()
        if len(score) == 0:
            return 0.0
        else:
            score = score[0].strip()
            float_score = float(score.split()[2][:-1])
            return float_score
