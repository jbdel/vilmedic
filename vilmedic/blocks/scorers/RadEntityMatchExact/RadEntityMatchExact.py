import numpy as np
import torch.nn as nn
import stanza
from stanza import Pipeline

import logging

logging.getLogger("stanza").setLevel(logging.WARNING)


class RadEntityMatchExact(nn.Module):
    def __init__(self):
        super().__init__()
        self.ner = Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                            **{'tokenize_batch_size': 256, 'ner_batch_size': 256})
        self.target_types = {'S-ANATOMY', 'S-OBSERVATION'}

    def forward(self, refs, hyps):
        docs_h = self.ner([stanza.Document([], text=d.lower().replace(' .', '.')) for d in hyps])
        docs_r = self.ner([stanza.Document([], text=d.lower().replace(' .', '.')) for d in refs])

        scores_e = []
        for doc_h, doc_r in zip(docs_h, docs_r):

            # NER
            ner_h = []
            for sentence in doc_h.sentences:
                ner_h.extend([ner['text'] for ner in sentence.to_dict() if ner['ner'] in self.target_types])

            ner_r = []
            for sentence in doc_r.sentences:
                ner_r.extend([ner['text'] for ner in sentence.to_dict() if ner['ner'] in self.target_types])

            # precision
            match_p = sum([1.0 for ner in ner_h if ner in ner_r])
            total_p = len(ner_h)
            pr_e = match_p / total_p if total_p > 0 else 0.0

            # recall
            match_r = sum([1.0 for ner in ner_h if ner in ner_r])
            total_r = len(ner_r)
            rc_e = match_r / total_r if total_r > 0 else 0.0

            # harmonic mean
            score_e = 2 * pr_e * rc_e / (pr_e + rc_e) if pr_e > 0.0 and rc_e > 0.0 else 0.0
            scores_e.append(score_e)

        mean_exact_e = np.mean(scores_e)
        return mean_exact_e, scores_e, docs_h, docs_r


if __name__ == '__main__':
    v = RadEntityMatchExact()
    x = v(hyps=['No pleural effusion. Normal heart size.', 'Normal heart size.'] * 1,
          refs=['No pleural effusions.', 'Enlarged heart.'] * 1)

    print(x[0], x[1])  # 0.7 [0.4, 1.0]
