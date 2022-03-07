# We have four types of outputs to leverage from radgraph
# 1. Entity token
# 2. Entity label
# 3. Relation tokens
# 4. Relation label

# Question for Jean Benoit: good idea to be based on f1 score ?
# We are using sets: so we do not account for repetitions of a word ! Except through relations ?

# We do three rewards for each type, so a total of six rewards
# +1 reward that does everything at the same time (maybe not that great)
# We combine these rewards with chosen coefficients (hyperparameters)

# To test our reward, we want that no token < token with wrong label < token with correct label
# Compares important clinical tokens of both texts
# Ex: is "pleural effusion" in both texts ?


# To be tested alone
def exact_entity_token_match_reward(hypothesis_annotation_list,
                                    reference_annotation_list):
    hypothesis_entity_token_list = set(
        map(lambda x: x['tokens'],
            hypothesis_annotation_list['entities'].values()))
    reference_entity_token_list = set(
        map(lambda x: x['tokens'],
            reference_annotation_list['entities'].values()))

    precision = sum([
        1 for x in hypothesis_entity_token_list
        if x in reference_entity_token_list
    ]) / len(hypothesis_entity_token_list)
    recall = sum([
        1 for x in reference_entity_token_list
        if x in hypothesis_entity_token_list
    ]) / len(reference_entity_token_list)
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


# Compares important clinical entities and their syntaxtic use in both text
# Ex: is "pleural effusion" as a Definitively Absent Observation in both texts ?
# To be tested alone
def exact_entity_token_and_label_match_reward(hypothesis_annotation_list,
                                              reference_annotation_list):
    hypothesis_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            hypothesis_annotation_list['entities'].values()))
    reference_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            reference_annotation_list['entities'].values()))

    precision = sum([
        1 for x in hypothesis_entity_token_and_label_list
        if x in reference_entity_token_and_label_list
    ]) / len(hypothesis_entity_token_and_label_list)
    recall = sum([
        1 for x in reference_entity_token_and_label_list
        if x in hypothesis_entity_token_and_label_list
    ]) / len(reference_entity_token_and_label_list)
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


# Ex: if "pleural effusion" is in both texts, is it as a Definitively Absent Observation in both texts?
# Not to be tested
def exact_entity_label_if_correct_token_match_reward(
        hypothesis_annotation_list, reference_annotation_list):

    hypothesis_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            hypothesis_annotation_list['entities'].values()))
    reference_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            reference_annotation_list['entities'].values()))

    hypothesis_entity_token_list = set(
        map(lambda x: x['tokens'],
            hypothesis_annotations['entities'].values()))
    reference_entity_token_list = set(
        map(lambda x: x['tokens'], reference_annotations['entities'].values()))

    hypothesis_entity_token_and_label_list = [
        (tokens, label)
        for (tokens, label) in hypothesis_entity_token_and_label_list
        if tokens in reference_entity_token_list
    ]
    reference_entity_token_and_label_list = [
        (tokens, label)
        for (tokens, label) in reference_entity_token_and_label_list
        if tokens in hypothesis_entity_token_list
    ]

    precision = sum([
        1 for x in hypothesis_entity_token_and_label_list
        if x in reference_entity_token_and_label_list
    ]) / len(hypothesis_entity_token_and_label_list)

    recall = sum([
        1 for x in reference_entity_token_and_label_list
        if x in hypothesis_entity_token_and_label_list
    ]) / len(reference_entity_token_and_label_list)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


# Test alone, maybe best one ?
def partially_exact_entity_token_and_label_match_reward(
        hypothesis_annotation_list, reference_annotation_list):

    hypothesis_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            hypothesis_annotation_list['entities'].values()))
    reference_entity_token_and_label_list = set(
        map(lambda x: (x['tokens'], x['label']),
            reference_annotation_list['entities'].values()))

    hypothesis_entity_token_list = set(
        map(lambda x: x['tokens'],
            hypothesis_annotation_list['entities'].values()))
    reference_entity_token_list = set(
        map(lambda x: x['tokens'],
            reference_annotation_list['entities'].values()))

    precision = sum([
        1 for (x, y) in hypothesis_entity_token_and_label_list if
        (x, y) in reference_entity_token_and_label_list
    ] + [
        0.5 for (x, y) in hypothesis_entity_token_and_label_list if (
            ((x, y) not in reference_entity_token_and_label_list) and
            (x in reference_entity_token_list))
    ]) / len(hypothesis_entity_token_and_label_list)

    recall = sum([
        1 for (x, y) in reference_entity_token_and_label_list if
        (x, y) in hypothesis_entity_token_and_label_list
    ] + [
        0.5 for (x, y) in reference_entity_token_and_label_list if (
            ((x, y) not in hypothesis_entity_token_and_label_list) and
            (x in hypothesis_entity_token_list))
    ]) / len(reference_entity_token_and_label_list)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


# We do not do the partially exact loss for the relations, because does not make sense to do not have the label
# Just an exact match one that we combine with the other reward, and we are good

# Compares the relations, directly both the start and end tokens AS WELL AS the label
# Because a relation without the label does not really make sense


def exact_relation_token_and_label_match_reward(hypothesis_annotation_list,
                                                reference_annotation_list):
    hypothesis_relation_token_and_label_list = list(
        map(
            lambda x: [(x['tokens'], relation[0], relation[1])
                       for relation in x['relations']],
            hypothesis_annotation_list['entities'].values()))

    hypothesis_relation_token_and_label_list = set([
        (tokens_1, label,
         hypothesis_annotation_list['entities'][tokens_2]['tokens'])
        for relations in hypothesis_relation_token_and_label_list
        for (tokens_1, label, tokens_2) in relations
    ])

    reference_relation_token_and_label_list = list(
        map(
            lambda x: [(x['tokens'], relation[0], relation[1])
                       for relation in x['relations']],
            reference_annotation_list['entities'].values()))

    reference_relation_token_and_label_list = set([
        (tokens_1, label,
         reference_annotation_list['entities'][tokens_2]['tokens'])
        for relations in reference_relation_token_and_label_list
        for (tokens_1, label, tokens_2) in relations
    ])

    precision = sum([
        1 for x in hypothesis_relation_token_and_label_list
        if x in reference_relation_token_and_label_list
    ]) / len(hypothesis_relation_token_and_label_list)

    recall = sum([
        1 for x in reference_relation_token_and_label_list
        if x in hypothesis_relation_token_and_label_list
    ]) / len(reference_relation_token_and_label_list)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def partially_exact_relation_token_and_label_match_reward(
        hypothesis_annotation_list, reference_annotation_list):

    hypothesis_relation_token_and_label_list = list(
        map(
            lambda x: [(x['tokens'], relation[0], relation[1])
                       for relation in x['relations']],
            hypothesis_annotation_list['entities'].values()))

    hypothesis_relation_token_and_label_list = set([
        (tokens_1, label,
         hypothesis_annotation_list['entities'][tokens_2]['tokens'])
        for relations in hypothesis_relation_token_and_label_list
        for (tokens_1, label, tokens_2) in relations
    ])

    hypothesis_relation_token_list = set([
        (token_1, token_2)
        for (token_1, label,
             token_2) in hypothesis_relation_token_and_label_list
    ])

    reference_relation_token_and_label_list = list(
        map(
            lambda x: [(x['tokens'], relation[0], relation[1])
                       for relation in x['relations']],
            reference_annotation_list['entities'].values()))

    reference_relation_token_and_label_list = set([
        (tokens_1, label,
         reference_annotation_list['entities'][tokens_2]['tokens'])
        for relations in reference_relation_token_and_label_list
        for (tokens_1, label, tokens_2) in relations
    ])

    reference_relation_token_list = set([
        (token_1, token_2)
        for (token_1, label,
             token_2) in reference_relation_token_and_label_list
    ])

    precision = sum([
        1 for (x, y, z) in hypothesis_relation_token_and_label_list if
        (x, y, z) in reference_relation_token_and_label_list
    ] + [
        0.5 for (x, y, z) in hypothesis_relation_token_and_label_list if ((
            (x, y, z) not in reference_relation_token_and_label_list) and (
                (x, z) in reference_relation_token_list))
    ]) / len(hypothesis_relation_token_and_label_list)

    recall = sum([
        1 for (x, y, z) in reference_relation_token_and_label_list if
        (x, y, z) in hypothesis_relation_token_and_label_list
    ] + [
        0.5 for (x, y, z) in reference_relation_token_and_label_list if ((
            (x, y, z) not in hypothesis_relation_token_and_label_list) and (
                (x, z) in hypothesis_relation_token_list))
    ]) / len(reference_relation_token_and_label_list)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


hypothesis_annotations = {
    'text':
    'FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process .',
    'entities': {
        '1': {
            'tokens': 'lungs',
            'label': 'ANAT-DP',
            'start_ix': 28,
            'end_ix': 28,
            'relations': []
        },
        '2': {
            'tokens': 'clear',
            'label': 'OBS-DP',
            'start_ix': 30,
            'end_ix': 30,
            'relations': [['located_at', '1']]
        },
        '3': {
            'tokens': 'focal',
            'label': 'OBS-DA',
            'start_ix': 32,
            'end_ix': 32,
            'relations': [['modify', '4']]
        },
        '4': {
            'tokens': 'consolidation',
            'label': 'OBS-DA',
            'start_ix': 33,
            'end_ix': 33,
            'relations': [['located_at', '1']]
        },
        '5': {
            'tokens': 'edema',
            'label': 'OBS-DA',
            'start_ix': 37,
            'end_ix': 37,
            'relations': []
        },
        '6': {
            'tokens': 'cardiomediastinal',
            'label': 'ANAT-DP',
            'start_ix': 40,
            'end_ix': 40,
            'relations': []
        },
        '7': {
            'tokens': 'silhouette',
            'label': 'ANAT-DP',
            'start_ix': 41,
            'end_ix': 41,
            'relations': [['modify', '6']]
        },
        '8': {
            'tokens': 'within',
            'label': 'OBS-DP',
            'start_ix': 43,
            'end_ix': 43,
            'relations': []
        },
        '9': {
            'tokens': 'normal',
            'label': 'OBS-DP',
            'start_ix': 44,
            'end_ix': 44,
            'relations': [['located_at', '6']]
        },
        '10': {
            'tokens': 'limits',
            'label': 'OBS-DP',
            'start_ix': 45,
            'end_ix': 45,
            'relations': [['modify', '9']]
        },
        '11': {
            'tokens': 'acute',
            'label': 'OBS-DA',
            'start_ix': 48,
            'end_ix': 48,
            'relations': [['modify', '13']]
        },
        '12': {
            'tokens': 'osseous',
            'label': 'ANAT-DP',
            'start_ix': 49,
            'end_ix': 49,
            'relations': []
        },
        '13': {
            'tokens': 'abnormalities',
            'label': 'OBS-DA',
            'start_ix': 50,
            'end_ix': 50,
            'relations': [['located_at', '12']]
        },
        '14': {
            'tokens': 'acute',
            'label': 'OBS-DA',
            'start_ix': 55,
            'end_ix': 55,
            'relations': [['modify', '16']]
        },
        '15': {
            'tokens': 'cardiopulmonary',
            'label': 'ANAT-DP',
            'start_ix': 56,
            'end_ix': 56,
            'relations': []
        },
        '16': {
            'tokens': 'process',
            'label': 'OBS-DA',
            'start_ix': 57,
            'end_ix': 57,
            'relations': [['located_at', '15']]
        }
    },
    'data_source': None,
    'data_split': 'inference'
}
reference_annotations = {
    'text':
    'FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process .',
    'entities': {
        '1': {
            'tokens': 'lungs',
            'label': 'ANAT-DP',
            'start_ix': 28,
            'end_ix': 28,
            'relations': []
        },
        '2': {
            'tokens': 'clear',
            'label': 'OBS-DP',
            'start_ix': 30,
            'end_ix': 30,
            'relations': [['located_at', '1']]
        },
        '3': {
            'tokens': 'focal',
            'label': 'OBS-DA',
            'start_ix': 32,
            'end_ix': 32,
            'relations': [['modify', '4']]
        },
        '4': {
            'tokens': 'consolidation',
            'label': 'OBS-DA',
            'start_ix': 33,
            'end_ix': 33,
            'relations': [['located_at', '1']]
        },
        '5': {
            'tokens': 'edema',
            'label': 'OBS-DA',
            'start_ix': 37,
            'end_ix': 37,
            'relations': []
        },
        '6': {
            'tokens': 'cardiomediastinal',
            'label': 'ANAT-DP',
            'start_ix': 40,
            'end_ix': 40,
            'relations': []
        },
        '7': {
            'tokens': 'silhouette',
            'label': 'ANAT-DP',
            'start_ix': 41,
            'end_ix': 41,
            'relations': [['modify', '6']]
        },
        '8': {
            'tokens': 'within',
            'label': 'OBS-DP',
            'start_ix': 43,
            'end_ix': 43,
            'relations': []
        },
        '9': {
            'tokens': 'normal',
            'label': 'OBS-DP',
            'start_ix': 44,
            'end_ix': 44,
            'relations': [['located_at', '6']]
        },
        '10': {
            'tokens': 'limits',
            'label': 'OBS-DP',
            'start_ix': 45,
            'end_ix': 45,
            'relations': [['modify', '9']]
        },
        '11': {
            'tokens': 'acute',
            'label': 'OBS-DA',
            'start_ix': 48,
            'end_ix': 48,
            'relations': [['modify', '13']]
        },
        '12': {
            'tokens': 'osseous',
            'label': 'ANAT-DP',
            'start_ix': 49,
            'end_ix': 49,
            'relations': []
        },
        '13': {
            'tokens': 'abnormalities',
            'label': 'OBS-DA',
            'start_ix': 50,
            'end_ix': 50,
            'relations': [['located_at', '12']]
        },
        '14': {
            'tokens': 'acute',
            'label': 'OBS-DA',
            'start_ix': 55,
            'end_ix': 55,
            'relations': [['modify', '16']]
        },
        '15': {
            'tokens': 'cardiopulmonary',
            'label': 'ANAT-DP',
            'start_ix': 56,
            'end_ix': 56,
            'relations': []
        },
        '16': {
            'tokens': 'process',
            'label': 'OBS-DA',
            'start_ix': 57,
            'end_ix': 57,
            'relations': [['located_at', '15']]
        }
    },
    'data_source': None,
    'data_split': 'inference'
}

#score = partially_exact_relation_token_and_label_match_reward(
#    hypothesis_annotations, reference_annotations)

#print(score)
#print(2 * (13 / 15) * (13 / 14) / (13 / 15 + 13 / 14))
