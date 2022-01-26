import re
import os
import argparse
import csv
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = os.path.join(__file__.split('vilmedic/data/')[0], 'vilmedic/data/')

RULES = {
    'rrg': ['findings', 'impression', 'one_image_per_study'],
    'rrs': ['impression_and_findings'],
    'selfup': ['impression_and_or_findings'],
}

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
args = parser.parse_args()


def _open(str, mode):
    path = os.path.join(DATA_PATH, args.task.upper(), "mimic-cxr")
    file = os.path.join(path, str)
    os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
    return open(file, mode)


def main():
    try:
        rules = RULES[args.task]
    except KeyError:
        raise KeyError("task {} does not exist.".format(args.task))

    print('#### mimic_cxr_sectioned.csv')
    reports = defaultdict(dict)
    skipped_reports_i = 0
    skipped_reports_f = 0
    skipped_reports_i_f = 0

    for row in tqdm(csv.DictReader(open('mimic_cxr_sectioned.csv'))):

        impression = re.sub("\s+", " ", row['impression'])  # removing all line breaks
        findings = re.sub("\s+", " ", row['findings'])  # removing all line breaks

        if not impression:
            skipped_reports_i += 1

        if not findings:
            skipped_reports_f += 1

        if not impression and not findings:
            skipped_reports_i_f += 1

        if impression:
            reports['impression'][row['study']] = impression
        if findings:
            reports['findings'][row['study']] = findings

    print('len(reports) impression', len(reports['impression']))
    print('len(reports) findings', len(reports['findings']))

    print('No impression', skipped_reports_i)
    print('No findings', skipped_reports_f)
    print('No impression + findings', skipped_reports_i_f)
    #

    print('####  mimic-cxr-2.0.0-splits.csv')

    for r in rules:
        if r in ["impression", "findings", "impression_and_findings", "impression_and_or_findings"]:

            if r == "impression_and_findings":
                files = {
                    "train_impression": _open("train.impression.tok", "w"),
                    "test_impression": _open("test.impression.tok", "w"),
                    "validate_impression": _open("validate.impression.tok", "w"),
                    "train_findings": _open("train.findings.tok", "w"),
                    "test_findings": _open("test.findings.tok", "w"),
                    "validate_findings": _open("validate.findings.tok", "w"),
                    "train_image": _open("train.image.tok", "w"),
                    "test_image": _open("test.image.tok", "w"),
                    "validate_image": _open("validate.image.tok", "w"),
                }

            if r == "impression_and_or_findings":
                files = {
                    "train_report": _open("train.report.tok", "w"),
                    "test_report": _open("test.report.tok", "w"),
                    "validate_report": _open("validate.report.tok", "w"),
                    "train_image": _open("train.image.tok", "w"),
                    "test_image": _open("test.image.tok", "w"),
                    "validate_image": _open("validate.image.tok", "w"),
                }

            if r == "impression":
                files = {
                    "train_impression": _open("impression/train.impression.tok", "w"),
                    "test_impression": _open("impression/test.impression.tok", "w"),
                    "validate_impression": _open("impression/validate.impression.tok", "w"),
                    "train_image": _open("impression/train.image.tok", "w"),
                    "test_image": _open("impression/test.image.tok", "w"),
                    "validate_image": _open("impression/validate.image.tok", "w"),
                }

            if r == "findings":
                files = {
                    "train_findings": _open("findings/train.findings.tok", "w"),
                    "test_findings": _open("findings/test.findings.tok", "w"),
                    "validate_findings": _open("findings/validate.findings.tok", "w"),
                    "train_image": _open("findings/train.image.tok", "w"),
                    "test_image": _open("findings/test.image.tok", "w"),
                    "validate_image": _open("findings/validate.image.tok", "w"),
                }

            skipped_study = 0
            written_study = defaultdict(int)
            study_set = set()

            for row in tqdm(csv.DictReader(open('mimic-cxr-2.0.0-split.csv'))):

                study_id = 's' + row['study_id']
                if "one_image_per_study" in rules and study_id in study_set:
                    continue

                im_path = os.path.join(
                    'mimic-cxr-images-512/files',
                    'p' + str(row['subject_id'])[:2],  # 10000032 -> p10
                    'p' + str(row['subject_id']),
                    's' + str(row['study_id']),
                    row['dicom_id'] + '.jpg'
                )

                if r == "impression":
                    if study_id not in reports["impression"]:
                        skipped_study += 1
                        continue
                    files[row['split'] + '_impression'].write(reports["impression"][study_id] + '\n')

                if r == "findings":
                    if study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    files[row['split'] + '_findings'].write(reports["findings"][study_id] + '\n')

                if r == "impression_and_findings":
                    if study_id not in reports["impression"] and study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    files[row['split'] + '_impression'].write(reports["impression"][study_id] + '\n')
                    files[row['split'] + '_findings'].write(reports["findings"][study_id] + '\n')

                if r == "impression_and_or_findings":
                    if study_id not in reports["impression"] or study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    if study_id not in reports["impression"]:
                        files[row['split'] + '_report'].write(reports["findings"][study_id] + '\n')
                    if study_id not in reports["findings"]:
                        files[row['split'] + '_report'].write(reports["impression"][study_id] + '\n')
                    if study_id in reports["impression"] or study_id in reports["findings"]:
                        files[row['split'] + '_report'].write(
                            reports["impression"][study_id] + ' ' + reports["findings"][study_id] + '\n')

                files[row['split'] + '_image'].write(im_path + '\n')
                written_study[row['split']] += 1
                if "one_image_per_study" in rules:
                    study_set.add(study_id)

            print('rule', r, 'skipped studies because no reports', skipped_study)
            print('rule', r, 'written studies', written_study)

    [v.close() for k, v in files.items()]


main()
