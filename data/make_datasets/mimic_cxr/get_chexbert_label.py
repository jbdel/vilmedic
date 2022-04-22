import os
import sys
import tqdm
import argparse

from vilmedic.blocks.scorers import CheXbert

LABELS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
          "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
          "Support Devices", "No Finding"]


def compute_labels(report_file):
    dir_path = os.path.dirname(os.path.realpath(report_file))
    report_file_name = os.path.basename(report_file)

    # get split
    chunks = report_file_name.split('.')
    assert len(chunks) == 3, "input file name must be of type 'split.report.tok'"
    split = chunks[0]
    ext = chunks[2]

    # output file
    output_file = os.path.join(dir_path, '{}.label.{}'.format(split, ext))
    print("The labels will be written in {}".format(output_file))

    # Compute labels
    reports = open(report_file).readlines()
    chexbert = CheXbert()
    chexbert_output = [chexbert.get_label(r.strip(), mode="classification") for r in
                       tqdm.tqdm(reports, total=len(reports))]

    split_labels = []
    for label in chexbert_output:
        new_label = []
        for i, v in enumerate(label):
            if isinstance(v, int) and v > 0:
                new_label.append(LABELS[i])
        if not new_label:
            new_label = ["No Finding"]

        split_labels.append(new_label)

    open(output_file, "w").write("\n".join([",".join([c for c in l]) for l in split_labels]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report_file', type=str)
    args, _ = parser.parse_known_args()
    compute_labels(args.report_file)
