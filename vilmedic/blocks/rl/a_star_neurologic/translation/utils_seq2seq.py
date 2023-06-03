

def tokenize_constraints(tokenizer, raw_cts):
    def tokenize(phrase):
        token_ids = [tokenizer.encoder.get(x) for x in tokenizer.spm_target.EncodeAsPieces(phrase)]
        if phrase.startswith(' ('):
            token_ids = token_ids[1:]
        assert all([x is not None for x in token_ids]), f'unrecognized token in {phrase} {type}'
        return token_ids, True
    return [[list(map(tokenize, clause)) for clause in ct] for ct in raw_cts]
