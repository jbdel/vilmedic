**News**
| Papers  | | 
| ----------- | ----------- |
|  Toward Expanding the Scope of Radiology Report Summarization to Multiple Anatomies and Modalities | [Dataset](https://vilmedic.app/papers/acl2023) 
|  Overview of the RadSum23 Shared Task on Multi-modal and Multi-anatomical Radiology Report Summarization | [Challenge](https://vilmedic.app/misc/bionlp23/sharedtask/) 
|  Improving the Factual Correctness of Radiology Report Generation with Semantic Rewards | [Replicate](https://vilmedic.app/papers/emnlp2022/) 

---

### ViLMedic: a framework for research at the intersection of vision and language in medical AI

<p align="center">
  <img src="https://vilmedic.app/favicon/favicon-64x64.png" alt="" style="width: 14px;"> ViLMedic has a dedicated website at: <a href="https://vilmedic.app/">https://vilmedic.app/</a>
</p>

<p align="center">
  <img src="vilmedic/logo.png" width="190px">
  <br />
  <br />
   <a href="https://github.com/jbdel/vilmedic/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <img src="https://img.shields.io/badge/Stanford-Medicine-red" />
</p>

---

```bibtex
@inproceedings{delbrouck-etal-2022-vilmedic,
    title = "{V}i{LM}edic: a framework for research at the intersection of vision and language in medical {AI}",
    author = "Delbrouck, Jean-benoit  and
      Saab, Khaled  and
      Varma, Maya  and
      Eyuboglu, Sabri  and
      Chambon, Pierre  and
      Dunnmon, Jared  and
      Zambrano, Juan  and
      Chaudhari, Akshay  and
      Langlotz, Curtis",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-demo.3",
    pages = "23--34",
}
```


# Quickstart and documentation

<p align="center">
Rendez-vous at: <a href="https://vilmedic.app/installation/">https://vilmedic.app/installation/</a>
</p>





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


#### Blocks

| Blocks  | 
| ----------- | 
| **Natural Language Processing**
| HuggingFace transformer encoder and decoder
| HuggingFace transformer beam-search and model ensembling :fire:	
| NLG metrics (BLEU, ROUGE, METEOR, MAUVE) and Radiology Reports Generation metrics ([F1-CheXbert](https://github.com/stanfordmlgroup/CheXbert))
| [RadGraph](https://openreview.net/pdf?id=pMWtc5NKd7V)
| **Vision**
| All PyTorch VisualEncoder architectures 
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


## Citation

If you use ViLMedic in your work or use any models published in ViLMedic, please cite:
```
@inproceedings{delbrouck-etal-2022-vilmedic,
    title = "{V}i{LM}edic: a framework for research at the intersection of vision and language in medical {AI}",
    author = "Delbrouck, Jean-benoit  and
      Saab, Khaled  and
      Varma, Maya  and
      Eyuboglu, Sabri  and
      Chambon, Pierre  and
      Dunnmon, Jared  and
      Zambrano, Juan  and
      Chaudhari, Akshay  and
      Langlotz, Curtis",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-demo.3",
    pages = "23--34",
}
```
## License
ViLMedic is MIT-licensed. The license applies to the pre-trained models as well.
