import os
import glob
import json
import re
import argparse

import os, sys
from vilmedic.constants import EXTRA_CACHE_DIR

from .reward_functions import *

model_path = os.path.join(EXTRA_CACHE_DIR, "radgraph.tar.gz")
data_path = os.path.join(os.path.dirname(__file__), "data/")
out_path = os.path.join(os.path.dirname(__file__), "result.json")


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

    with open("./temp_dygie_input.json", 'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")


def run_inference(model_path, cuda):
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path = "./temp_dygie_output.json"
    data_path = "./temp_dygie_input.json"

    os.system(f"allennlp predict {model_path} {data_path} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --silent")


def postprocess_reports():
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    file_name = f"./temp_dygie_output.json"
    data = []

    with open(file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))

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


def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    os.system("rm temp_dygie_input.json")
    os.system("rm temp_dygie_output.json")


def run(model_path, report_list, cuda=-1):
    preprocess_reports(report_list)

    run_inference(model_path, cuda)

    final_dict = postprocess_reports()

    cleanup()

    return final_dict


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


# simple, partial, complete
# 1. simple is exact_entity_token_match_reward
# 2. partial is partially_exact_entity_token_and_label_match_reward
# and partially_exact_relation_token_and_label_match_reward
# 3. complete is exact_entity_token_and_label_match_reward
# and exact_relation_token_and_label_match_reward
def get_rewards(hypothesis_report_list,
                reference_report_list,
                lambda_e=0.5,
                lambda_r=0.5,
                reward_level="partial",
                cuda=-1):
    global model_path, out_path

    number_of_reports = len(hypothesis_report_list)
    assert (len(reference_report_list) == number_of_reports)

    # We do not want to run radgraph on empty reports
    empty_report_index_list = [
        i for i in range(number_of_reports)
        if (len(hypothesis_report_list[i]) == 0) or (
                len(reference_report_list[i]) == 0)
    ]

    number_of_non_empty_reports = number_of_reports - len(
        empty_report_index_list)

    report_list = [
                      hypothesis_report
                      for i, hypothesis_report in enumerate(hypothesis_report_list)
                      if i not in empty_report_index_list
                  ] + [
                      reference_report
                      for i, reference_report in enumerate(reference_report_list)
                      if i not in empty_report_index_list
                  ]

    inference_dict = run(model_path, report_list, cuda)

    reward_list = []
    hypothesis_annotation_lists = []
    reference_annotation_lists = []

    non_empty_report_index = 0
    for report_index in range(number_of_reports):
        if report_index in empty_report_index_list:
            reward_list.append(0)
            continue

        hypothesis_annotation_list = inference_dict[str(
            non_empty_report_index)]
        reference_annotation_list = inference_dict[str(
            non_empty_report_index + number_of_non_empty_reports)]

        reward_list.append(
            compute_reward(hypothesis_annotation_list,
                           reference_annotation_list, lambda_e, lambda_r,
                           reward_level))

        reference_annotation_lists.append(reference_annotation_list)
        hypothesis_annotation_lists.append(hypothesis_annotation_list)
        non_empty_report_index += 1

    return reward_list, hypothesis_annotation_lists, reference_annotation_lists

# get_rewards(report_list, report_list)
