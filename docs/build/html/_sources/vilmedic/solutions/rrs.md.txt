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
	<b>Data requirements: </b> images and RRS data
	<br/>Please refer to <a href="https://vilmedic.readthedocs.io/en/latest/vilmedic/solutions/data.html">the data section</a>.
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

| Dataset | ROUGE-2   |  ROUGEL | F1-cheXbert (micro-avg) | F1-cheXbert-5 (micro-avg) | Config
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **mimic-cxr-validate**
| <span class="card" id="rrs_biomed-roberta-mimic-val">BioMed-RoBERTa</span> (single model) | 48.15 | 57.34  |  77.54  |  75.24  | [RRS/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)
| **mimic-cxr-test**
| <span class="card" id="rrs_biomed-roberta-mimic-test">BioMed-RoBERTa</span> (single model) | 34.09 | 45.98  |  72.71  |  74.64  |  [RRS/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)
| **indiana-validate**
| <span class="card" id="rrs_biomed-roberta-indiana-val">BioMed-RoBERTa</span> (single model) | 70.19 | 76.92  |  86.59 |  71.15 | [RRS/biomed-roberta-baseline-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-indiana.yml)
| **indiana-test**
| <span class="card" id="rrs_biomed-roberta-indiana-test">BioMed-RoBERTa</span> (single model) | 69.85 | 77.42  |  85.10 |  70.68 | [RRS/biomed-roberta-baseline-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-indiana.yml)
