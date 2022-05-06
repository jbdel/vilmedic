# Filename: ciderD.py
#
# Description: Describes the class to compute the CIDEr-D (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from .ciderD_RL_scorer import CiderScorer


class CiderDRL:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, n=4, sigma=6.0, df="corpus"):
        # set cider to sum over 1 to 4-grams
        super().__init__()
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set which where to compute document frequencies from
        refs = [ref.strip() for ref in open(df).readlines()]
        scorer = CiderScorer(refs=refs)
        scorer.compute_doc_freq()
        self._df = scorer.document_frequency

    def __call__(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma, df=self._df)
        res = {i: [v] for i, v in enumerate(res)}
        gts = {i: [v] for i, v in enumerate(gts)}
        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)
            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr-D"


if __name__ == '__main__':
    import pickle

    x = CiderDRL(df="../../../../../data/RRG/mimic-cxr/impression/train.impression.tok")
    for i in range(200000):
        x(
            gts=[
                'no evidence of consolidation to suggest pneumonia is seen. there  is some retrocardiac atelectasis. a small left pleural effusion may be  present. no pneumothorax is seen. no pulmonary edema. a right granuloma is  unchanged. the heart is mildly enlarged, unchanged. there is tortuosity of  the aorta.',
                'there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.'
            ],
            res=[
                'heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
                'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.'
            ])
