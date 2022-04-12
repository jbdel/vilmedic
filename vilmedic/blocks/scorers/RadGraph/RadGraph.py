import os
import torch.nn as nn
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from allennlp.commands.predict import _predict, _PredictManager
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.checks import check_for_gpu
from .utils import preprocess_reports, postprocess_reports, compute_reward

from vilmedic.constants import EXTRA_CACHE_DIR

import logging
logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("tqdm").setLevel(logging.CRITICAL)


class RadGraph(nn.Module):
    def __init__(self,
                 lambda_e=0.5,
                 lambda_r=0.5,
                 reward_level="partial",
                 batch_size=1,
                 cuda=0):

        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_r = lambda_r
        self.reward_level = reward_level
        self.cuda = cuda
        self.batch_size = batch_size
        self.model_path = os.path.join(EXTRA_CACHE_DIR, "radgraph.tar.gz")

        # Model
        import_plugins()
        import_module_and_submodules("dygie")

        check_for_gpu(self.cuda)
        archive = load_archive(
            self.model_path,
            weights_file=None,
            cuda_device=self.cuda,
            overrides='',
        )
        self.predictor = Predictor.from_archive(
            archive, predictor_name='dygie', dataset_reader_to_load='validation'
        )

    def forward(self, refs, hyps):
        # Preprocessing
        number_of_reports = len(hyps)
        empty_report_index_list = [
            i for i in range(number_of_reports)
            if (len(hyps[i]) == 0) or (
                    len(refs[i]) == 0)
        ]

        number_of_non_empty_reports = number_of_reports - len(
            empty_report_index_list)

        report_list = [hypothesis_report
                       for i, hypothesis_report in enumerate(hyps)
                       if i not in empty_report_index_list
                       ] + [reference_report
                            for i, reference_report in enumerate(refs)
                            if i not in empty_report_index_list
                            ]

        model_input = preprocess_reports(report_list)
        # AllenNLP
        manager = _PredictManager(
            predictor=self.predictor,
            input_file=str(model_input),  # trick the manager, make the list as string so it thinks its a filename
            output_file=None,
            batch_size=self.batch_size,
            print_to_console=False,
            has_dataset_reader=True,
        )
        results = manager.run()

        # Postprocessing
        inference_dict = postprocess_reports(results)

        # Compute reward
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
                compute_reward(hypothesis_annotation_list, reference_annotation_list, self.lambda_e, self.lambda_r,
                               self.reward_level))

            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        return np.mean(reward_list), reward_list, hypothesis_annotation_lists, reference_annotation_lists


if __name__ == '__main__':
    m = RadGraph()
    report = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    hypothesis_report_list = [report, "", "a", report]

    report_2 = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The heart is clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    reference_report_list = [report_2, report_2, report_2, report_2]

    reward_list = m(hyps=hypothesis_report_list, refs=reference_report_list)
    print(reward_list[1])  # [0.8666666666666667, 0, 0, 0.8666666666666667]
