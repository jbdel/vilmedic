import os
import numpy as np
import torch.nn as nn
from vilmedic.blocks.scorers import RadEntityMatchExact
from vilmedic.blocks.scorers.RadEntityNLI.nli import SimpleNLI
from vilmedic.constants import EXTRA_CACHE_DIR


def _nli_label(prediction):
    best_label, best_prob = 'entailment', 0.0
    for label, prob in prediction.items():
        if prob > best_prob:
            best_label = label
            best_prob = prob
    return best_label


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

    def forward(self, refs, hyps):
        # Getting radiology reports with rad entities
        _, _, docs_h, docs_r = self.match_exact(refs, hyps)

        # Split reports as a list of sentences
        refs_reports_split = []
        hyps_reports_split = []
        for doc_h, doc_r in zip(docs_h, docs_r):
            hyps_reports_split.append([' '.join([token['text'] for token in sentence.to_dict()]) for sentence in
                                       doc_h.sentences])
            refs_reports_split.append([' '.join([token['text'] for token in sentence.to_dict()]) for sentence in
                                       doc_r.sentences])

        # Getting BERT scores
        _, _, _, stats = self.nli.sentence_scores_bert_score(refs_reports_split, hyps_reports_split,
                                                             label='all',
                                                             prf='f')
        hypos_nli, refs_nli = [], []
        for i, stat in enumerate(stats):
            ref_stat = stat['scores'][0]
            hyp_stat = stat['scores'][1]
            refs_nli.append({sid: _nli_label(pred[0]) for sid, pred in ref_stat.items()})
            hypos_nli.append({sid: _nli_label(pred[0]) for sid, pred in hyp_stat.items()})

        assert len(docs_h) == len(docs_r) == len(refs_nli) == len(hypos_nli)

        # Compute NLI score
        scores_e = []
        for doc_h, doc_r, hypo_nli, ref_nli in zip(docs_h, docs_r, hypos_nli, refs_nli):

            # NER, with sentence number
            ner_h = []
            for i, sentence in enumerate(doc_h.sentences):
                ner_h.extend([(i, ner['text']) for ner in sentence.to_dict() if ner['ner'] in self.target_types])

            ner_r = []
            for i, sentence in enumerate(doc_r.sentences):
                ner_r.extend([(i, ner['text']) for ner in sentence.to_dict() if ner['ner'] in self.target_types])

            # precision
            total_p = len(ner_h)
            match_p = 0
            entity_ner_r = [ent[1] for ent in ner_r]
            for index, entity in ner_h:
                if hypo_nli[index] == 'contradiction':
                    continue
                if hypo_nli[index] == 'entailment':
                    match_p += 1
                    continue
                if hypo_nli[index] == 'neutral' and entity in entity_ner_r:
                    match_p += 1
            pr_e = match_p / total_p if total_p > 0 else 0.0

            # recall
            total_r = len(ner_r)
            match_r = 0
            entity_ner_h = [ent[1] for ent in ner_h]
            for index, entity in ner_r:
                if ref_nli[index] == 'contradiction':
                    continue
                if ref_nli[index] == 'entailment':
                    match_r += 1
                    continue
                if ref_nli[index] == 'neutral' and entity in entity_ner_h:
                    match_r += 1
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
