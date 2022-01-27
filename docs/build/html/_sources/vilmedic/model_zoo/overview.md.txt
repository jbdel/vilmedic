# Overview

## Usage

``` 
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained(model_name)
batch = processor.inference(seq=["no acute cardiopulmonary process"],
                            image=["my_chest_xray.jpg"])
out = model(**batch)
```

## Models

| Name  |   dataset | Model Card | Report preprocessing
| ------------- |:-------------:|:-------------:|:-------------:|
| **Radiology report generation** 
| rrg/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| rrg/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/) | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| rrg/baseline-padchest| [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   | | -
| **Radiology report summarization** 
| rrs/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   | | [rouge](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L70)
| rrs/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| **Self-supervision** 
| selfsup/convirt-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-mimic-balanced | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-padchest-16 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   | | [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/convirt-padchest-32 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   | | [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/convirt-indiana-16 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-indiana-32 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-indiana-64 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)  | | [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| [selfsup/gloria-chexpert](https://github.com/marshuang80/gloria)  | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)   | | [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/gloria-mimic-48  | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/simclr-mimic-16 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-32 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-64 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/vae-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/vae-indiana | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)
| selfsup/vae-padchest | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/) 
| **Medical VQA** 
| mvqa/mvqa-imageclef| [ImageCLEF-VQAMed](https://www.imageclef.org/2021/medical/vqa)   
