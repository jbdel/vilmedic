<div class="warning_box">
	<b>Warning: </b> The models are resource-hungry. If you can't run a configuration because the training batch-size 
	is too big, you can use gradient accumulation as such:
	<div class="highlight">
<pre>python bin/train.py config/task/conf.yml \
    trainor.batch_size=8 \
    trainor.grad_accu=8     </pre></div>	
</div>


# Radiology Report Summarization

The automatic summarization of radiology reports has several clinical applications such as accelerating the radiology workflow and improving the efficiency of clinical communications.

This task aims to promote the development of clinical summarization models that are able to generate radiology impression statements by summarizing textual findings written by radiologists.


<div class="data_box">
	<b>Data requirements: </b> mimic-cxr-images and RRS data
	<div class="highlight">
<pre>python data/download.py mimic-cxr-images-512,RRS </pre></div>	
</div>



## BioMed-RoBERTa baseline 

### Models
The model is defined as such in the config file:
```
model:
  proto: RRS
  encoder:
    proto: data/RRS/huggingface/biomed_roberta_base
  decoder:
    proto: data/RRS/huggingface/biomed_roberta_base
```

### Metrics and scores

This task has no previous work on the main open datasets.

| Dataset |     ROUGE-L | F1-cheXbert (micro) | Config
| ------------- |:-------------:|:-------------:|:-------------:|
| **Mimic-validate**
| <span id="#rrs_biomed-roberta-mimic">BioMed-RoBERTa</span> (single model)  | 22.46  |  45.04  | [RRS/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)
| **Mimic-test**
| <span id="#rrs_biomed-roberta-mimic">BioMed-RoBERTa</span> (single model)  | 22.46  |  45.04  | [RRS/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)


