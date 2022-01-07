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
	<b>Data requirements: </b> mimic-cxr-images and RRG data
	<div class="highlight">
<pre>python data/download.py mimic-cxr-images-512,RRG </pre></div>	
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

| Dataset |     ROUGE-L | F1-cheXbert (micro) | Config
| ------------- |:-------------:|:-------------:|:-------------:|
| **Mimic-test**
| [M2-Trans (2021)](https://arxiv.org/pdf/2010.10042.pdf) |  -  |  44.70 |
| <span id="#rrg_biomed-roberta-mimic">BioMed-RoBERTa</span> (single model)   | 22.46  |  45.04  | [RRG/biomed-roberta-baseline-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)

