
def tokenize_constraints(tokenizer, raw_cts):
    def tokenize(phrase):
        tokens = tokenizer.tokenize(phrase)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids, True
    return [[list(map(tokenize, clause)) for clause in ct] for ct in raw_cts]