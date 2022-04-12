import os
import torch
import time
import numpy as np
import torch.nn as nn
from vilmedic.blocks.scorers import RadEntityMatchExact
from vilmedic.blocks.scorers.RadEntityNLI.nli import SimpleNLI
from vilmedic.constants import EXTRA_CACHE_DIR

from torchmetrics.functional.text.bert import BERTScorer
from itertools import chain, product

import logging

logging.getLogger("stanza").setLevel(logging.WARNING)


class RadEntityNLI(nn.Module):
    def __init__(self):
        super().__init__()
        # NER types
        self.target_types = {'S-ANATOMY', 'S-OBSERVATION'}
        self.match_exact = RadEntityMatchExact()

        # NLI scorer
        model = SimpleNLI.load_model(os.path.join(EXTRA_CACHE_DIR, "model_medrad_19k.gz"))
        self.nli = SimpleNLI(model, batch=24, neutral_score=0.3333333333333333, nthreads=2, pin_memory=False,
                             bert_score='distilbert-base-uncased', cache=200000,
                             verbose=False)

        # BertSore
        self.bert_scorer = BERTScorer(model_type='distilbert-base-uncased',
                                      num_layers=5,
                                      batch_size=64,
                                      nthreads=4,
                                      all_layers=False,
                                      idf=False,
                                      device='cuda',
                                      lang='en',
                                      rescale_with_baseline=True,
                                      baseline_path=None)

    def forward(self, refs, hyps):
        t = time.time()
        with torch.no_grad():
            # # Getting radiology reports with rad entities
            _, _, docs_h, docs_r = self.match_exact(refs, hyps)

            scores_e = []
            for doc_h, doc_r in zip(docs_h, docs_r):

                hyp_report = [' '.join([token['text'] for token in sentence.to_dict()]) for sentence in
                              doc_h.sentences]
                ref_report = [' '.join([token['text'] for token in sentence.to_dict()]) for sentence in
                              doc_r.sentences]

                if len(hyp_report) == 0 or len(ref_report) == 0:
                    continue

                ner_h = [[ner['text'] for ner in sentence.to_dict() if ner['ner'] in self.target_types] for sentence in
                         doc_h.sentences]
                ner_r = [[ner['text'] for ner in sentence.to_dict() if ner['ner'] in self.target_types] for sentence in
                         doc_r.sentences]

                # getting all sentence pairs scores
                pairs = list(product(hyp_report, ref_report))
                _, _, f_scores = self.bert_scorer.score(
                    cands=[p[0] for p in pairs],
                    refs=[p[1] for p in pairs],
                    verbose=False,
                    batch_size=64,
                )

                f_scores = torch.reshape(torch.tensor(f_scores), (len(hyp_report), len(ref_report)))

                # self.bert_scorer.plot_example([p[0] for p in pairs][0], [p[1] for p in pairs][0])

                # precision
                match_p = 0
                entity_ner_r = list(chain.from_iterable(ner_r))
                total_p = 0
                for hyp_sentence, hyp_sentence_entities, hyp_f_score in zip(hyp_report, ner_h, f_scores):
                    # No entites in current sentence
                    if not hyp_sentence_entities:
                        continue
                    sim_index = torch.argmax(hyp_f_score)
                    nli_label = self.nli.predict([hyp_sentence], [ref_report[sim_index]])[1][0]
                    for entity in hyp_sentence_entities:
                        total_p += 1
                        if nli_label == 'contradiction':
                            continue
                        if nli_label == 'entailment':
                            match_p += 1
                            continue
                        if nli_label == 'neutral' and entity in entity_ner_r:
                            match_p += 1

                match_r = 0
                entity_ner_h = list(chain.from_iterable(ner_h))
                total_r = 0
                for ref_sentence, ref_sentence_entities, ref_f_score in zip(ref_report, ner_r, f_scores.T):
                    # No entites in current sentence
                    if not ref_sentence_entities:
                        continue
                    sim_index = torch.argmax(ref_f_score)
                    nli_label = self.nli.predict([ref_sentence], [hyp_report[sim_index]])[1][0]
                    for entity in ref_sentence_entities:
                        total_r += 1
                        if nli_label == 'contradiction':
                            continue
                        if nli_label == 'entailment':
                            match_r += 1
                            continue
                        if nli_label == 'neutral' and entity in entity_ner_h:
                            match_r += 1

                pr_e = match_p / total_p if total_p > 0 else 0.0
                rc_e = match_r / total_r if total_r > 0 else 0.0

                # harmonic mean
                score_e = 2 * pr_e * rc_e / (pr_e + rc_e) if pr_e > 0.0 and rc_e > 0.0 else 0.0
                scores_e.append(score_e)

            mean_exact_e = np.mean(scores_e)
            return mean_exact_e, scores_e


if __name__ == '__main__':
    x = RadEntityNLI()(
        refs=[
            'no evidence of consolidation to suggest pneumonia is seen. there  is some retrocardiac atelectasis. a small left pleural effusion may be  present. no pneumothorax is seen. no pulmonary edema. a right granuloma is  unchanged. the heart is mildly enlarged, unchanged. there is tortuosity of  the aorta.',
            'there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.'
        ],
        hyps=[
            'heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
            'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.'
        ])

    print(x)
