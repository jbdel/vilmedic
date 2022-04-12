import numpy as np
# 
# nli_model = SimpleNLI(model, batch=self.nli_batch, neutral_score=self.nli_neutral_score,
#                                       nthreads=self.nthreads, pin_memory=self.pin_memory,
#                                       bert_score=self.BERT_SCORE_DEFAULT, sentsplitter='linebreak',
#                                       cache=self.nli_cache, verbose=self.verbose)


def score(self, rids, hypos):
    # Named entity recognition
    hypo_sents = {}
    hypos_entities = {}
    texts, buf = [], []
    for hypo in hypos:
        buf.append(hypo)
        if len(buf) >= self.batch:
            text = '\n\n{0}\n\n'.format(self.DOC_SEPARATOR).join(buf)
            texts.append(text)
            buf = []
    if len(buf) > 0:
        text = '\n\n{0}\n\n'.format(self.DOC_SEPARATOR).join(buf)
        texts.append(text)
    i = 0
    for text in texts:
        doc = self.ner(text)
        j = 0
        for sentence in doc.sentences:
            if i not in hypos_entities:
                hypos_entities[i] = {}
            if i not in hypo_sents:
                hypo_sents[i] = ''
            if sentence.text == self.DOC_SEPARATOR:
                i += 1
                j = 0
            else:
                if len(hypo_sents[i]) > 0:
                    hypo_sents[i] += '\n'
                hypo_sents[i] += sentence.text
                for entity in sentence.ents:
                    if entity.type in self.target_types:
                        buf = []
                        for word in entity.words:
                            buf.append(word.text.lower())
                        s = ' '.join(buf)
                        if s not in hypos_entities:
                            hypos_entities[i][s] = [j]
                        else:
                            hypos_entities[i][s].append(j)
                j += 1
        i += 1
    hypo_nli, ref_nli = None, None
    if self.mode.startswith(self.MODE_NLI):
        hypo_nli, ref_nli = {}, {}
        texts1, texts2 = [], []
        for i, rid in enumerate(rids):
            buf = []
            rid = rid.split(self.ID_SEPARATOR)[0]
            for sid in sorted(self.sentences[rid].keys()):
                buf.append(self.sentences[rid][sid])
            texts1.append('\n'.join(buf))
            texts2.append(hypo_sents[i])
        _, _, _, stats = self.nli.sentence_scores_bert_score(texts1, texts2, label='all', prf=self.prf)
        for i in range(len(rids)):
            rid, rs = rids[i], stats[i]
            ref_nli[i] = {}
            for sid, tup in rs['scores'][0].items():
                pred, _ = self._nli_label(tup[0])
                ref_nli[i][sid] = pred
            hypo_nli[i] = {}
            for sid, tup in rs['scores'][1].items():
                pred, _ = self._nli_label(tup[0])
                hypo_nli[i][sid] = pred
    # Calculate scores
    scores_e, scores_n = [], []
    for i, rid in enumerate(rids):
        hypo_entities = hypos_entities[i]
        rid = rid.split(self.ID_SEPARATOR)[0]
        ref_entities = self.entities[rid]
        # precision
        match_e, match_n, total_pr = 0, 0, 0
        if self.prf != 'r':
            for s in hypo_entities.keys():
                for sid in hypo_entities[s]:
                    if s in ref_entities:
                        match_e += 1
                        if hypo_nli is None:
                            match_n += 1.0
                    if hypo_nli is not None:
                        if hypo_nli[i][sid] == 'neutral':
                            if s in ref_entities:
                                match_n += 1.0
                        elif hypo_nli[i][sid] == 'entailment':
                            if s in hypo_entities:
                                match_n += 1.0
                            else:
                                if self.mode == self.MODE_NLI or self.mode == self.MODE_NLI_ENTAILMENT:
                                    match_n += self.entail_score
                        elif hypo_nli[i][sid] == 'contradiction':
                            if self.mode == self.MODE_NLI_ENTAILMENT:
                                if s in hypo_entities:
                                    match_n += 1.0
                    total_pr += 1
        pr_e = match_e / total_pr if total_pr > 0 else 0.0
        pr_n = match_n / total_pr if total_pr > 0 else 0.0
        # recall
        match_e, match_n, total_rc = 0, 0, 0
        if self.prf != 'p':
            for s in ref_entities.keys():
                for sid in ref_entities[s]:
                    if s in hypo_entities:
                        match_e += 1
                        if ref_nli is None:
                            match_n += 1.0
                    if ref_nli is not None:
                        if ref_nli[i][sid] == 'neutral':
                            if s in hypo_entities:
                                match_n += 1.0
                        elif ref_nli[i][sid] == 'entailment':
                            if s in hypo_entities:
                                match_n += 1.0
                            else:
                                if self.mode == self.MODE_NLI or self.mode == self.MODE_NLI_ENTAILMENT:
                                    match_n += self.entail_score
                        elif ref_nli[i][sid] == 'contradiction':
                            if self.mode == self.MODE_NLI_ENTAILMENT:
                                if s in hypo_entities:
                                    match_n += 1.0
                    total_rc += 1
        rc_e = match_e / total_rc if total_rc > 0 else 0.0
        rc_n = match_n / total_rc if total_rc > 0 else 0.0
        # fb1
        if self.prf == 'p':
            score_e, score_n = pr_e, pr_n
        elif self.prf == 'r':
            score_e, score_n = rc_e, rc_n
        else:
            score_e = 2 * pr_e * rc_e / (pr_e + rc_e) if pr_e > 0.0 and rc_e > 0.0 else 0.0
            score_n = 2 * pr_n * rc_n / (pr_n + rc_n) if pr_n > 0.0 and rc_n > 0.0 else 0.0
        if self.penalty:
            penalty = np.e ** (-((total_pr - total_rc) ** 2) / (2 * self.PENALTY_SIGMA ** 2))
            score_e *= penalty
            score_n *= penalty
        scores_e.append(score_e)
        scores_n.append(score_n)
    mean_exact_e = np.mean(scores_e)
    mean_exact_n = np.mean(scores_n)
    return mean_exact_e, scores_e, mean_exact_n, scores_n
