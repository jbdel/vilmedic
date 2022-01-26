# Data


<div class="summary_box">
	<b>Content:</b><br/>
	<ol>
	<li>	<a href="#mimic-cxr">MIMIC-CXR</a>	</li>
	<li>	<a href="#padchest">PadChest</a>	</li>
	<li>	<a href="#indiana-university">Indiana-University</a>	</li>
	<li>	<a href="#chexpert">CheXpert</a>	</li>
	<li>	<a href="#imageclef-vqa">Imageclef-VQA</a>	</li>	</ol>
</div>

## MIMIC-CXR
[[README.MD]](https://github.com/jbdel/vilmedic/tree/main/data/make_datasets/mimic_cxr)
## PadChest
[[README.MD]](https://github.com/jbdel/vilmedic/tree/main/data/make_datasets/padchest)
## Indiana-University
[[Official Link]](https://www.kaggle.com/raddar/chest-xrays-indiana-university)

Indiana-University is available in ViLMedic using `vilmedic-download` in your shell. Type this command to prompt the list of available downloads.

For example, if you want to download Indiana-University for the RRG task, type:

```
vilmedic-download indiana-images-512,RRG
```

## CheXpert
[[Official Link]](https://stanfordmlgroup.github.io/competitions/chexpert/)

You can download CheXpert images for inference on our model-zoo using the official link. 
Since the reports are not available, you cannot train multimodal models using ViLMedic.


## Imageclef-VQA
[[Official Link]](https://github.com/abachaa/VQA-Med-2021)

Imageclef-VQA is available in ViLMedic using `vilmedic-download` in your shell. Type this command to prompt the list of available downloads.

For example, if you want to download Imageclef-VQA for the Medical-VQA task, type:

```
vilmedic-download imageclef-vqa-images-512,RRG
```
