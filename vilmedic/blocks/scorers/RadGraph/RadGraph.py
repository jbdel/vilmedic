import os
import torch.nn as nn
import numpy as np
import sys
import logging

from vilmedic.constants import EXTRA_CACHE_DIR
from vilmedic.zoo.utils import download_model
from vilmedic.blocks.scorers.RadGraph.utils import (
    preprocess_reports,
    postprocess_reports,
    compute_reward,
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
#
logging.getLogger("allennlp").setLevel(logging.CRITICAL)
logging.getLogger("tqdm").setLevel(logging.CRITICAL)
logging.getLogger("filelock").setLevel(logging.CRITICAL)

from allennlp.commands.predict import _predict, _PredictManager
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.checks import check_for_gpu


class RadGraph(nn.Module):
    def __init__(
            self,
            lambda_e=0.5,
            lambda_r=0.5,
            reward_level="full",
            batch_size=1,
            cuda=0,
            **kwargs
    ):

        super().__init__()
        assert reward_level in ["simple", "complete", "partial", "full"]
        self.lambda_e = lambda_e
        self.lambda_r = lambda_r
        self.reward_level = reward_level
        self.cuda = cuda
        self.batch_size = batch_size

        self.model_path = os.path.join(EXTRA_CACHE_DIR, "radgraph.tar.gz")

        if not os.path.exists(self.model_path):
            download_model(
                repo_id="StanfordAIMI/RRG_scorers",
                cache_dir=EXTRA_CACHE_DIR,
                filename="radgraph.tar.gz",
            )

        # Model
        import_plugins()
        import_module_and_submodules("dygie")

        check_for_gpu(self.cuda)
        archive = load_archive(
            self.model_path,
            weights_file=None,
            cuda_device=self.cuda,
            overrides="",
        )
        self.predictor = Predictor.from_archive(
            archive, predictor_name="dygie", dataset_reader_to_load="validation"
        )

    def forward(self, refs, hyps):
        # Preprocessing
        number_of_reports = len(hyps)

        assert len(refs) == len(hyps)

        empty_report_index_list = [
            i
            for i in range(number_of_reports)
            if (len(hyps[i]) == 0) or (len(refs[i]) == 0)
        ]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # import pickle
        # if os.path.exists("./temp"):
        #     inference_dict = pickle.load(open("./temp", "rb"))
        # else:
        model_input = preprocess_reports(report_list)
        # AllenNLP
        manager = _PredictManager(
            predictor=self.predictor,
            input_file=str(
                model_input
            ),  # trick the manager, make the list as string so it thinks its a filename
            output_file=None,
            batch_size=self.batch_size,
            print_to_console=False,
            has_dataset_reader=True,
        )
        results = manager.run()

        # Postprocessing
        inference_dict = postprocess_reports(results)
        # pickle.dump(inference_dict, open("./temp", "wb"))

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                if self.reward_level == "full":
                    reward_list.append((0., 0., 0.))
                else:
                    reward_list.append(0.)

                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.lambda_e,
                    self.lambda_r,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports

        if self.reward_level == "full":
            reward_list_ = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            reward_list = reward_list_
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward = np.mean(reward_list)

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


if __name__ == "__main__":
    import time

    m = RadGraph(cuda=0, reward_level="partial", batch_size=1)
    # report = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    # hypothesis_report_list = [report, "", "a", report]
    #
    # report_2 = "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The heart is clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    # reference_report_list = [report_2, report_2, report_2, report_2]
    #
    # reward_list = m(hyps=hypothesis_report_list, refs=reference_report_list)
    t = time.time()
    num = str(103276)
    l1 = open("test_best-1_881942_hyps.txt").readlines()
    # l1 = [l.strip() for l in l1][:10]
    l1 = [l.strip() for l in l1]
    l2 = open("test_best-1_103276_refs.txt").readlines()
    # l2 = [l.strip() for l in l2][:10]
    l2 = [l.strip() for l in l2]
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = m(hyps=l1, refs=l2)
    # print(time.time() - t)
    print(mean_reward)  # [0.8666666666666667, 0, 0, 0.8666666666666667]


# ^[(0.353946348023485, 0.32697070866071776, 0.25986992412367665)

