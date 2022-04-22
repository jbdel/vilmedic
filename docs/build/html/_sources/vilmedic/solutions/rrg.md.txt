<div class="warning_box">
	<b>Warning: </b> The models are resource-hungry. If you can't run a configuration because the training batch-size 
	is too big, you can use gradient accumulation as such:
	<div class="highlight">
<pre>python bin/train.py config/task/conf.yml \
    trainor.batch_size=8 \
    trainor.grad_accu=8     </pre></div>	
</div>


# Radiology Report Generation

An important new application of natural language generation (NLG) is to build assistive systems that take X-ray images of a patient and generate a textual report describing clinical observations in the images.

<div class="data_box">
	<b>Data requirements: </b> images and RRG data
	<br/>Please refer to <a href="https://vilmedic.readthedocs.io/en/latest/vilmedic/solutions/data.html">the data section</a>.
</div>



## BioMed-RoBERTa baseline 

### Models
The model is defined as such in the config file:
```
model:
  proto: RRG
  decoder:
    proto: data/RRG/huggingface/biomed_roberta_base
  cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features
    dropout_out: 0.0
    permute: batch_first
    visual_embedding_dim: 1664
    freeze: False
```

### Metrics and scores

| Dataset |     ROUGEL |  F1-cheXbert (micro) | F1-cheXbert-5 (micro) | Config
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| **mimic-cxr-validation**
| <span class="card" id="rrg_biomed-roberta-mimic-val">BioMed-RoBERTa</span> (single model)   | 32.72  |  49.12  | 43.10  | [RRG/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)
| **mimic-cxr-test**
| [M2-Trans (2021)](https://arxiv.org/pdf/2010.10042.pdf) |  -  |  - | 44.70 |
| <span class="card" id="rrg_biomed-roberta-mimic-test">BioMed-RoBERTa</span> (single model)   | 22.45  |  44.23  |  45.08  | [RRG/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)
| **indiana-validation**
| <span class="card" id="rrg_biomed-roberta-indiana-val">BioMed-RoBERTa</span> (single model)   | 20.78  |  61.96  |  34.07  | [RRG/biomed-roberta-baseline-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-indiana.yml)
| **indiana-test**
| [M2-Trans (2021)](https://arxiv.org/pdf/2010.10042.pdf) |  -  |  - | 32.20 |
| <span id="rrg_biomed-roberta-indiana-test">BioMed-RoBERTa</span> (single model)   | 21.41  |  60.58  |  31.04  | [RRG/biomed-roberta-baseline-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-indiana.yml)
| **padchest-validation**
| <span class="card" id="baseline_padchest-val">baseline</span> (single model)   | 16.54  |  -  |  -  | [RRG/baseline-padchest.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/baseline-padchest.yml)
| **padchest-test**
| <span class="card" id="baseline_padchest-test">baseline</span> (single model)   | 16.32  |  -  |  -  | [RRG/baseline-padchest.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/baseline-padchest.yml)

