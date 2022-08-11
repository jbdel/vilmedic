import re
import os
import argparse
import csv
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = os.path.join(__file__.split('vilmedic/data/')[0], 'vilmedic/data/')

DICOM_VIEWS = {row["dicom_id"]: row["ViewPosition"] for row in csv.DictReader(open("mimic-cxr-2.0.0-metadata.csv"))}

RULES = {
    'rrg': ['findings', 'impression'],
    'rrs': ['impression_and_findings'],
    'selfup': ['impression_and_or_findings'],
}

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
args = parser.parse_args()


def reorder_images(im_paths):
    dicoms = [os.path.basename(im).replace('.jpg', '') for im in im_paths]
    views = [DICOM_VIEWS[d] for d in dicoms]
    ranked_views = ['PA', 'AP', 'LATERAL', 'LL', 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA RLD', 'PA LLD', 'LAO', 'RAO',
                    'LPO', 'XTABLE LATERAL', 'SWIMMERS', '']
    reorder_path = []
    for r in ranked_views:
        for i, v in enumerate(views):
            if r == v:
                reorder_path.append(im_paths[i])
    return reorder_path


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

            # Grouping images per study_id
            study_images = defaultdict(list)
            for row in tqdm(csv.DictReader(open('mimic-cxr-2.0.0-split.csv'))):
                key = ('s' + row['study_id'], row['split'])
                study_images[key].append(os.path.join(
                    'mimic-cxr-images-512/files',
                    'p' + str(row['subject_id'])[:2],  # 10000032 -> p10
                    'p' + str(row['subject_id']),
                    's' + str(row['study_id']),
                    row['dicom_id'] + '.jpg'
                ))

            for key, im_paths in study_images.items():
                study_id, split = key
                if r == "impression":
                    if study_id not in reports["impression"]:
                        skipped_study += 1
                        continue
                    files[split + '_impression'].write(reports["impression"][study_id] + '\n')

                if r == "findings":
                    if study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    files[split + '_findings'].write(reports["findings"][study_id] + '\n')

                if r == "impression_and_findings":
                    if study_id not in reports["impression"] and study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    files[split + '_impression'].write(reports["impression"][study_id] + '\n')
                    files[split + '_findings'].write(reports["findings"][study_id] + '\n')

                if r == "impression_and_or_findings":
                    if study_id not in reports["impression"] or study_id not in reports["findings"]:
                        skipped_study += 1
                        continue
                    if study_id not in reports["impression"]:
                        files[split + '_report'].write(reports["findings"][study_id] + '\n')
                    if study_id not in reports["findings"]:
                        files[split + '_report'].write(reports["impression"][study_id] + '\n')
                    if study_id in reports["impression"] or study_id in reports["findings"]:
                        files[split + '_report'].write(
                            reports["impression"][study_id] + ' ' + reports["findings"][study_id] + '\n')

                im_paths = reorder_images(im_paths)
                if "one_image_per_study" in rules:
                    files[split + '_image'].write(im_paths[0] + '\n')
                else:
                    files[split + '_image'].write(','.join(im_paths) + '\n')

                written_study[split] += 1

            print('rule', r, ': skipped studies because no reports', skipped_study)
            print('rule', r, ': written studies', written_study)

    [v.close() for k, v in files.items()]


main()
