import gzip
import os
import site
import sys
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, OrderedDict
from bert_score.utils import bert_cos_score_idf, cache_scibert, get_idf_dict, get_model, lang2model, model2layers
from cachetools import LRUCache
from torch.nn import DataParallel
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
from .BERTNLI import BERTNLI


class NLTKTokenizer:
    @staticmethod
    def tokenize(text):
        return wordpunct_tokenize(text)


class _NLIScorer:
    LABEL_ALL = 'all'
    LABEL_CONTRADICT = 'contradiction'
    LABEL_ENTAIL = 'entailment'
    LABEL_NEUTRAL = 'neutral'
    METRIC_FB1 = 'f'
    PENALTY_SIGMA = 6.0
    THRESH_FIX = 'fix'
    THRESH_NONE = 'none'

    def __init__(self, neutral_score=(1.0 / 3), batch=16, nthreads=2, pin_memory=False, bert_score=None,
                 cache=None, verbose=False):
        self.model = None
        self.neutral_score = neutral_score
        self.batch = batch
        self.nthreads = nthreads
        self.pin_memory = pin_memory

        self.tokenizer = NLTKTokenizer()
        if bert_score is not None:
            self.bert_score_model = BERTScorer(model_type=bert_score, batch_size=batch, nthreads=nthreads,
                                               lang='en', rescale_with_baseline=True)
        else:
            self.bert_score_model = None
        self.cache = LRUCache(cache) if cache is not None else None
        self.verbose = verbose
        self.gpu = False

    def cuda(self):
        self.model = self.model.cuda()
        self.gpu = True
        if self.bert_score_model is not None:
            self.bert_score_model = self.bert_score_model.cuda()
        return self

    def predict(self, premises, hypotheses):
        raise NotImplementedError

    def sentence_scores_bert_score(self, texts1, texts2, label='entailment', thresh='none', prf='f'):
        # Calculate BertScores
        tids, tsents1, tsents2, bsents1, bsents2 = [], {}, {}, [], []
        bsents1t, bsents2t, thresh_nums = [], [], [0, 0]
        for tid, (text1, text2) in enumerate(zip(texts1, texts2)):
            tids.append(tid)
            sents1 = text1
            tsents1[tid] = sents1
            sents2 = text2
            tsents2[tid] = sents2
            for i, sent1 in enumerate(sents1):
                for j, sent2 in enumerate(sents2):
                    bsents1.append(sent1)
                    bsents2.append(sent2)
        bsents1 += bsents1t
        bsents2 += bsents2t
        # Store BERTScores to dictionaries of (#text, #sentence)
        _, _, bf = self.bert_score_model.score(bsents1, bsents2)
        bf = bf.numpy()
        idx = 0
        scores1, scores2 = {}, {}
        for tid in tids:
            scores1[tid], scores2[tid] = {}, {}
            for i in range(len(tsents1[tid])):
                scores1[tid][i] = []
                for j in range(len(tsents2[tid])):
                    if j not in scores2[tid]:
                        scores2[tid][j] = []
                    score = bf[idx]
                    scores1[tid][i].append(score)
                    scores2[tid][j].append(score)
                    idx += 1
        thresh1, thresh2 = -100.0, -100.0
        # Obtain high BertScore premise-hypothesis pairs
        tids_all, prems_all, hypos_all = [], [], []
        prems, hypos, pbfs, fidxs = {}, {}, {}, {}
        valid_sents, cache_sents = {}, {}
        for k in [0, 1]:
            prems[k], hypos[k], pbfs[k], fidxs[k] = {}, {}, {}, {}
            valid_sents[k], cache_sents[k] = {}, {}
        for tid in tids:
            for k, quad in enumerate([(scores1, tsents1, tsents2, thresh1), (scores2, tsents2, tsents1, thresh2)]):
                # ref sent: prem=gen & hypo=ref, gen sent: prem=ref & hypo=gen
                scores, sents_hypos, sents_prems, thresh = quad
                prems[k][tid], hypos[k][tid], pbfs[k][tid], fidxs[k][tid] = {}, {}, {}, {}
                valid_sents[k][tid], cache_sents[k][tid] = {}, {}
                for i in range(len(scores[tid])):
                    if len(scores[tid][i]) > 0:
                        fidx = np.argmax(scores[tid][i])
                        fscore = scores[tid][i][fidx]
                        # Align with the best scoring sentence
                        prems[k][tid][i] = sents_prems[tid][fidx]
                        hypos[k][tid][i] = sents_hypos[tid][i]
                        pbfs[k][tid][i] = fscore
                        fidxs[k][tid][i] = fidx
                        # Set IDs, premises, and hypothesis for NLI
                        prem = sents_prems[tid][fidx]
                        hypo = sents_hypos[tid][i]
                        if self.cache is not None and (prem, hypo) in self.cache:
                            cache_sents[k][tid][i] = self.cache[(prem, hypo)]
                            valid_sents[k][tid][i] = None
                        else:
                            if scores[tid][i][fidx] >= thresh and ((k == 0 and prf != 'p') or (k == 1 and prf != 'r')):
                                tids_all.append(tid)
                                prems_all.append(prem)
                                hypos_all.append(hypo)
                                valid_sents[k][tid][i] = (prem, hypo)
                            else:
                                valid_sents[k][tid][i] = None

        # NLI probabilities
        probs_all, _ = self.predict(prems_all, hypos_all)
        probs_tids = OrderedDict()
        for i, tid in enumerate(tids_all):
            if tid not in probs_tids:
                probs_tids[tid] = []
            probs_tids[tid].append(probs_all[i])
        prs, rcs, fb1s, stats = [], [], [], []
        for tid in tids:
            rc_pr = {0: [], 1: []}
            sent_probs = [{}, {}]
            idx = 0

            for k, num_sent in zip([0, 1], (len(tsents1[tid]), len(tsents2[tid]))):
                for i in range(num_sent):
                    if i in cache_sents[k][tid]:
                        prob = cache_sents[k][tid][i]
                    elif i in valid_sents[k][tid] and valid_sents[k][tid][i] is not None:
                        prem_hypo = valid_sents[k][tid][i]
                        prob = probs_tids[tid][idx]
                        if self.cache is not None:
                            self.cache[prem_hypo] = prob
                        idx += 1
                    else:
                        prob = None

                    # Soft scoring
                    if prob is not None:
                        score = prob[self.LABEL_ENTAIL]
                        rc_pr[k].append(score)
                    else:
                        prob = {self.LABEL_ENTAIL: -1.0, self.LABEL_NEUTRAL: -1.0, self.LABEL_CONTRADICT: -1.0}
                    if i in prems[k][tid] and i in hypos[k][tid]:
                        sent_probs[k][i] = (prob, pbfs[k][tid][i], prems[k][tid][i], hypos[k][tid][i], fidxs[k][tid][i])
            mean_precision = np.mean(rc_pr[1]) if len(rc_pr[1]) > 0 else 0.0
            mean_recall = np.mean(rc_pr[0]) if len(rc_pr[0]) > 0 else 0.0
            if mean_precision + mean_recall > 0.0:
                fb1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            else:
                fb1 = 0.0
            prs.append(mean_precision)
            rcs.append(mean_recall)
            fb1s.append(fb1)
            stats.append({'scores': sent_probs, 'threshes': (thresh1, thresh2)})
        return prs, rcs, fb1s, stats

    def stop(self):
        pass


class BERTScorer:
    PENALTY_SIGMA = 6.0

    def __init__(self, refs=None, model_type=None, num_layers=None, verbose=False, idf=False, batch_size=16, nthreads=2,
                 all_layers=False, lang=None, rescale_with_baseline=False, penalty=False):
        assert lang is not None or model_type is not None, 'Either lang or model_type should be specified'
        if rescale_with_baseline:
            assert lang is not None, 'Need to specify Language when rescaling with baseline'

        if model_type is None:
            lang = lang.lower()
            model_type = lang2model[lang]
        if num_layers is None:
            num_layers = model2layers[model_type]

        if model_type.startswith('scibert'):
            tokenizer = AutoTokenizer.from_pretrained(cache_scibert(model_type))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = get_model(model_type, num_layers, all_layers)

        if not idf:
            idf_dict = defaultdict(lambda: 1.)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0
        elif isinstance(idf, dict):
            if verbose:
                print('using predefined IDF dict...')
            idf_dict = idf
        else:
            if verbose:
                print('preparing IDF dict...')
            start = time.perf_counter()
            idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
            if verbose:
                print('done in {:.2f} seconds'.format(time.perf_counter() - start))

        self.batch_size = batch_size
        self.verbose = verbose
        self.all_layers = all_layers
        self.penalty = penalty

        self.tokenizer = tokenizer
        self.model = model
        self.idf_dict = idf_dict
        self.device = 'cpu'

        self.baselines = None
        if rescale_with_baseline:
            baseline_path = None
            for sitepackage in site.getsitepackages():
                if baseline_path is None:
                    candidate_path = os.path.join(sitepackage, 'bert_score',
                                                  f'rescale_baseline/{lang}/{model_type}.tsv')
                    if os.path.exists(candidate_path):
                        baseline_path = candidate_path
            if baseline_path is not None and os.path.isfile(baseline_path):
                if not all_layers:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
                else:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()
                self.baselines = baselines
            else:
                print(f'Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}', file=sys.stderr)

    def cuda(self):
        self.device = 'cuda:0'
        self.model.cuda()
        return self

    def score(self, cands, refs):
        assert len(cands) == len(refs)

        if self.verbose:
            print('calculating scores...')
        start = time.perf_counter()
        all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict, verbose=self.verbose,
                                       device=self.device, batch_size=self.batch_size, all_layers=self.all_layers).cpu()
        if self.baselines is not None:
            all_preds = (all_preds - self.baselines) / (1 - self.baselines)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
        if self.penalty:
            for idx, (cand, ref) in enumerate(zip(cands, refs)):
                toks1 = self.tokenizer.tokenize(cand)
                toks2 = self.tokenizer.tokenize(ref)
                penalty = np.e ** (-((len(toks1) - len(toks2)) ** 2) / (2 * self.PENALTY_SIGMA ** 2))
                out[-1][idx] *= penalty

        if self.verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

        return out


class SimpleNLI(_NLIScorer):
    def __init__(self, model, neutral_score=(1.0 / 3), batch=16, nthreads=2, pin_memory=False, bert_score=None,
                 cache=None, verbose=False):
        super(SimpleNLI, self).__init__(neutral_score, batch, nthreads, pin_memory, bert_score, cache,
                                        verbose)

        self.model = DataParallel(model)
        self.model.cuda()
        self.model.eval()
        self.model.eval()

    @classmethod
    def load_model(cls, states=None):
        bert_type = 'bert'
        name = 'bert-base-uncased'
        bertnli = BERTNLI(name, bert_type=bert_type, length=384, force_lowercase=True, device='cpu')
        with gzip.open(states, 'rb') as f:
            states_dict = torch.load(f, map_location=torch.device('cpu'))
        bertnli.load_state_dict(states_dict, strict=False)
        return bertnli

    def predict(self, sent1s, sent2s):
        batches, buf1, buf2 = [], [], []
        for sent1, sent2 in zip(sent1s, sent2s):
            buf1.append(sent1)
            buf2.append(sent2)
            if len(buf1) >= self.batch:
                batches.append((buf1, buf2))
                buf1, buf2 = [], []
        if len(buf1) > 0:
            batches.append((buf1, buf2))

        probs, preds = [], []
        with torch.no_grad():
            for b1, b2 in batches:
                out = self.model(b1, b2)
                out = softmax(out, dim=-1).detach().cpu()
                _, idxs = out.max(dim=-1)
                for i, idx in enumerate(idxs):
                    idx = int(idx)
                    probs.append({'entailment': float(out[i][BERTNLI.LABEL_ENTAILMENT]),
                                  'neutral': float(out[i][BERTNLI.LABEL_NEUTRAL]),
                                  'contradiction': float(out[i][BERTNLI.LABEL_CONTRADICTION])})
                    if idx == BERTNLI.LABEL_ENTAILMENT:
                        preds.append('entailment')
                    elif idx == BERTNLI.LABEL_NEUTRAL:
                        preds.append('neutral')
                    elif idx == BERTNLI.LABEL_CONTRADICTION:
                        preds.append('contradiction')
                    else:
                        raise ValueError('Unknown label index {0}'.format(idx))
        return probs, preds
