
<p align="center">
  <img src="docs/logo.png" width="190px">
  <br />
  <br />
  <a href="https://vilmedic.readthedocs.io/en/latest/">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/vilmedic/badge/?version=latest"/>
  </a>
   <a href="https://github.com/jbdel/vilmedic/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <img src="https://img.shields.io/badge/Stanford-Medicine-red" />
</p>

---

ViLMedic: a framework for research at the intersection of vision and language in medical AI

## Installation
```
conda create --name vilmedic python==3.9 -y
git clone https://github.com/jbdel/vilmedic
python setup.py develop
```


## Documentation

Learn more about ViLMedic [here](https://vilmedic.readthedocs.io/en/latest/).

## Model Zoo

ViLMedic hosts a [zoo of pretrained models](https://vilmedic.readthedocs.io/en/latest/vilmedic/model_zoo/overview.html).

``` 
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/convirt-mimic")
batch = processor.inference(seq=["no acute cardiopulmonary process"],
                            image=["my_chest_xray.jpg"])

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'loss_l', 'loss_v', 'linguistic', 'visual'])
```

| Name  |   dataset | Report preprocessing
| ------------- |:-------------:|:-------------:|
| **Radiology report generation** 
| rrg/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| rrg/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/) |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| rrg/baseline-padchest| [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   |  -
| **Radiology report summarization** 
| rrs/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   |  [rouge](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L70)
| rrs/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| **Self-supervision** 
| selfsup/convirt-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-mimic-balanced | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-padchest-16 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   |  [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/convirt-padchest-32 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   |  [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/convirt-indiana-16 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-indiana-32 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/convirt-indiana-64 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)  |  [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| [selfsup/gloria-chexpert](https://github.com/marshuang80/gloria)  | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)   |  [gloria](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L34)
| selfsup/gloria-mimic-48  | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) |   [r2gen](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/base/papers/report_preprocessing.py#L6)
| selfsup/simclr-mimic-16 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-32 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-64 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/vae-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/vae-indiana | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)
| selfsup/vae-padchest | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/) 
| **Medical VQA** 
| mvqa/mvqa-imageclef| [ImageCLEF-VQAMed](https://www.imageclef.org/2021/medical/vqa)   



#### Implemented solutions

ViLMedic replicates solutions from the multimodal medical literature.

| Solutions  | 
| ----------- | 
| **Medical Visual Question Answering**
| [SYSU-HCP at VQA-Med 2021](http://ceur-ws.org/Vol-2936/paper-99.pdf)
| **Radiology report generation**
| [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf)
| [Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports](https://arxiv.org/abs/1911.02541)
| [Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation](https://arxiv.org/abs/2010.10042)
| **Radiology report summarization**
| [Multimodal Radiology Report Summarization](https://aclanthology.org/2021.bionlp-1.33/)
| **Multimodal self-supervised Learning**
| [Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://openreview.net/pdf?id=T4gXBOXoIUr)
| [DALLE: Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)
| [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
| [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
| [GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)

<!---    
#### Blocks

| Blocks  | 
| ----------- | 
| **Natural Language Processing**
| HuggingFace transformer encoder and decoder
| HuggingFace transformer beam-search and model ensembling :fire:	
| NLG metrics (BLEU, ROUGE, METEOR, MAUVE) and Radiology Reports Generation metrics ([F1-CheXbert](https://github.com/stanfordmlgroup/CheXbert))
| [RadGraph](https://openreview.net/pdf?id=pMWtc5NKd7V)
| **Vision**
| All PyTorch CNN architectures 
| [Vision Transformer](https://arxiv.org/abs/2010.11929)
| [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
| **Losses**
| All PyTorch losses
| ConVirt loss
| GLoRIA loss
| InfoNCE loss
| [SuperLoss](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf)
| **Reinforcement Learning**
| [Self-critical Sequence Training](https://arxiv.org/abs/1612.00563) (HuggingFace compliant) :fire:
| [PPO optimization](https://arxiv.org/abs/1612.00563)  (HuggingFace compliant)


-->

## Citation

If you use ViLMedic in your work or use any models published in ViLMedic, please cite:

```bibtex
@misc{Delbrouck2021ViLmedic,
  author =       {Jean-Benoit Delbrouck and Khaled Saab and Juan Manuel Zambrano Chaves and Pierre Joseph Marcel Chambon and Sabri Eyuboglu
 and Maya Varma and Jared Alexander Dunnmon and Curtis Langlotz and Akshay Chaudhari and Daniel Rubin},
  title =        {ViLMedic: A multimodal framework for vision and language medical research},
  howpublished = {\url{https://github.com/jbdel/vilmedic}},
  year =         {2021}
}
```

## License
ViLMedic is MIT-licensed. The license applies to the pre-trained models as well.