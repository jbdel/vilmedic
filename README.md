
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

ViLMedic (Vision-and-Language medical research) is a modular framework for multimodal research at the intersection of vision and language 
in the medical field. 

This framework contains reference implementations of state-of-the-art vision and language architectures, referred as "blocks" 
and full solutions for multimodal medical tasks using one or several blocks.

#### Implemented solutions
| Solution  | 
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


## Installation
```
conda create --name vilmedic python==3.9 -y
git clone https://github.com/jbdel/vilmedic
python setup.py develop
```


## Documentation

Learn more about ViLMedic [here](https://vilmedic.readthedocs.io/en/latest/).

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