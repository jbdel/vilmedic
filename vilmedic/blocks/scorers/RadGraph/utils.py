import json
import re
from .reward_functions import exact_entity_token_match_reward, partially_exact_relation_token_and_label_match_reward, \
    partially_exact_entity_token_and_label_match_reward, \
    exact_entity_token_and_label_match_reward, exact_relation_token_and_label_match_reward


def compute_reward(hypothesis_annotation_list, reference_annotation_list,
                   lambda_e, lambda_r, reward_level):
    if len(hypothesis_annotation_list["entities"].keys()) == 0 or len(
            reference_annotation_list["entities"].keys()) == 0:
        return 0

    if reward_level == "simple":
        return lambda_e * exact_entity_token_match_reward(
            hypothesis_annotation_list, reference_annotation_list)
    elif reward_level == "partial":
        return lambda_e * partially_exact_entity_token_and_label_match_reward(
            hypothesis_annotation_list, reference_annotation_list
        ) + lambda_r * partially_exact_relation_token_and_label_match_reward(
            hypothesis_annotation_list, reference_annotation_list)
    elif reward_level == "complete":
        return lambda_e * exact_entity_token_and_label_match_reward(
            hypothesis_annotation_list, reference_annotation_list
        ) + lambda_r * exact_relation_token_and_label_match_reward(
            hypothesis_annotation_list, reference_annotation_list)


def preprocess_reports(report_list):
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    final_list = []
    for idx, report in enumerate(report_list):
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',
                     report).split()
        temp_dict = {}

        temp_dict["doc_key"] = str(idx)

        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

    return [json.dumps(item) for item in final_list]


def postprocess_reports(results):
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}
    data = []

    for r in results:
        data.append(json.loads(r))

    for file in data:
        postprocess_individual_report(file, final_dict)

    return final_dict


def postprocess_individual_report(file, final_dict, data_source=None):
    """Postprocesses individual report

    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """

    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n, r, s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = "inference"

        final_dict[file['doc_key']] = temp_dict

    except:
        print(
            f"Error in doc key: {file['doc_key']}. Skipping inference on this file"
        )


def get_entity(n, r, s):
    """Gets the entities for individual reports

    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence

    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json

    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx + 1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [
            i for i, val in enumerate(rel_list) if val == [start_idx, end_idx]
        ]
        for i, val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab, str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx + 1)] = temp_dict

    return dict_entity
