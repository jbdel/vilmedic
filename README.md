
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
1. Medical Visual Question Answering
    1. SYSU-HCP at VQA-Med 2021 [[paper]](http://ceur-ws.org/Vol-2936/paper-99.pdf)
1. Radiology report generation
    1. Generating Radiology Reports via Memory-driven Transformer [[paper]](https://arxiv.org/pdf/2010.16056.pdf)
    1. Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports [[paper]](https://arxiv.org/abs/1911.02541)
    1. Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation [[paper]](https://arxiv.org/abs/2010.10042)
1. Radiology report summarization
    1. Multimodal Radiology Report Summarization [[paper]](https://aclanthology.org/2021.bionlp-1.33/)
1. Multimodal self-supervised Learning
    1. Contrastive Learning of Medical Visual Representations from Paired Images and Text [[paper]](https://openreview.net/pdf?id=T4gXBOXoIUr)
    1. DALLE: Zero-Shot Text-to-Image Generation [[paper]](https://arxiv.org/abs/2102.12092)
    1. CLIP: Learning Transferable Visual Models From Natural Language Supervision [[paper]](https://arxiv.org/abs/2103.00020)
    1. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations [[paper]](https://arxiv.org/abs/2002.05709)

    
#### Blocks
1. Natural Language Processing
    1. HuggingFace transformer encoder and decoder
    1. HuggingFace transformer beam-search and model ensembling :fire:	
    1. NLG metrics (BLEU, ROUGE, METEOR, MAUVE) and Radiology Reports Generation metrics ([F1-ChexBert](https://github.com/stanfordmlgroup/CheXbert),
     [RadGraph](https://openreview.net/pdf?id=pMWtc5NKd7V))
1. Vision
    1. All PyTorch CNN architectures 
    1. Vision Transformer [[paper]](https://arxiv.org/abs/2010.11929)
    1. [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
1. Losses
    1. All PyTorch losses
    1. ConVirt loss
    1. InfoNCE loss
    1. SuperLoss [[paper]](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf)
1. Reinforcement Learning
    1. Self-critical Sequence Training (HuggingFace compliant) :fire: [[paper]](https://arxiv.org/abs/1612.00563)
    1. PPO optimization (HuggingFace compliant) [[paper]](https://arxiv.org/abs/1612.00563)

## Model Zoo

ViLMedic hosts a [zoo of pretrained models](https://vilmedic.readthedocs.io/en/latest/vilmedic/model_zoo.html#).

``` 
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/convirt-mimic")
batch = processor.inference(seq=["no acute cardiopulmonary process"],
                            image=["files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"])

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'loss_l', 'loss_v', 'linguistic', 'visual'])
```


## Installation
```
git clone  https://github.com/jbdel/vilmedic
pip install -r requirements.txt
```


## Documentation

Learn more about ViLMedic [here](https://vilmedic.readthedocs.io/en/latest/).

## Citation

If you use ViLMedic in your work or use any models published in ViLMedic, please cite:

```bibtex
@misc{Delbrouck2021ViLmedic,
  author =       {Delbrouck, Jean-Benoit},
  title =        {ViLMedic: A multimodal framework for vision and language medical research},
  howpublished = {\url{https://github.com/jbdel/vilmedic}},
  year =         {2021}
}
```

## License
ViLMedic is MIT-licensed. The license applies to the pre-trained models as well.