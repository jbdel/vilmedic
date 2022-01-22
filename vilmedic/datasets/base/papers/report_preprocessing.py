import re
import six
from nltk.stem import porter
from nltk.tokenize import wordpunct_tokenize

# https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
# Generating Radiology Reports via Memory-driven Transformer
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


# https://github.com/ysmiura/ifcc/blob/6c111dbdfe7ce9d3150a5ad90360584cfd2b8442/clinicgen/text/tokenizer.py#L24
# Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation.
def ifcc_clean_report(report):
    report = report.lower()
    return ' '.join(wordpunct_tokenize(report))


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


# https://github.com/google-research/google-research/blob/master/rouge/tokenize.py
def rouge(text, use_stemmer=False):
    """Tokenize input text into a list of tokens.
    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.
    Args:
      text: A text blob to tokenize.
      use_stemmer: An optional stemmer.
    Returns:
      A list of string tokens extracted from input text.
    """

    # Pre-compile regexes that are use often
    NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
    NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
    SPACES_PATTERN = r"\s+"
    SPACES_RE = re.compile(SPACES_PATTERN)
    VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
    VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = NON_ALPHANUM_RE.sub(" ", six.ensure_str(text))

    tokens = SPACES_RE.split(text)
    if use_stemmer:
        stemmer = porter.PorterStemmer() if use_stemmer else None
        # Only stem words more than 3 characters long.
        tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x
                  for x in tokens]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]

    return tokens
