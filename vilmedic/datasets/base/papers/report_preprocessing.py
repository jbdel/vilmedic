import re


# https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
# https://arxiv.org/pdf/2010.16056.pdf
def r2gen_clean_report(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    if tokens == ['']:
        return ""
    report = ' . '.join(tokens) + ' .'
    return report


# https://github.com/ysmiura/ifcc/blob/master/extract_reports.py#L36
# Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation.
def ifcc_clean_report(report):
    space_pattern = re.compile('\\s+')
    report = report.replace('\n', ' ')
    return space_pattern.sub(' ', report)


# https://github.com/marshuang80/gloria/blob/main/gloria/models/gloria_model.py#L165
# GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition
def gloria_clean_report_chexpert(report):
    from nltk.tokenize import RegexpTokenizer
    t = report
    # use space instead of newline
    t = t.replace("\n", " ")

    # split sentences
    splitter = re.compile("[0-9]+\.")
    captions = splitter.split(t)
    captions = [point.split(".") for point in captions]
    captions = [sent for point in captions for sent in point]

    all_sents = []

    for t in captions:
        t = t.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(t.lower())

        if len(tokens) <= 1:
            continue

        included_tokens = []
        for t in tokens:
            t = t.encode("ascii", "ignore").decode("ascii")
            if len(t) > 0:
                included_tokens.append(t)
        all_sents.append(" ".join(included_tokens))

    t = " ".join(all_sents)
    return t
