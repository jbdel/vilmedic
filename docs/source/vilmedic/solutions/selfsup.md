# Self-supervision
Self-supervised learning (SSL) is a method of machine learning. It learns from unlabeled sample data. It can be regarded as an intermediate form between supervised and unsupervised learning.

<div class="data_box">
	<b>Data requirements: </b> mimic-cxr-images and SELFSUP data
	<div class="highlight">
<pre>python data/download.py mimic-cxr-images-512,SELFSUP </pre></div>	
</div>


## DALLE

[Zero-Shot Text-to-Image Generation](http://proceedings.mlr.press/v139/ramesh21a.html)


### Model
It is advised to use gradient accumulation. First, we need to train a VAE.

``` 
python bin/train.py config/CLIP/vae.yml \
    model.image_size=256 \
    model.num_layers=3 \
    model.num_tokens=8192 \
    model.codebook_dim=1024 \
    model.hidden_dim=64 \
    trainor.batch_size=16 \
    trainor.grad_accu=4 \
    name=vae
```



Secondly, we need to train a DALLE model using the trained VAE:
```
python bin/train.py config/CLIP/dalle.yml \
    model.vae.image_size=256 \
    model.dalle.dim=1024 \
    model.dalle.heads=16 \
    model.dalle.dim_head=64 \
    model.dalle.depth=16 \
    trainor.batch_size=12 \
    trainor.clip_grad_norm=0.5 \
    trainor.grad_accu=5 \
    trainor.lr_decay_params.patience=1 \
    model.vae.ckpt=data/CLIP/0.0022899999748915434_32_951498.pth \
    name="dalle" 
```     

### Metrics and scores

| Dataset  | Validation Loss | 
| ------------- |:-------------:|
| **mimic-cxr**   | 
| VAE   | 0.00345
| DALLE  | 1.6828  

### Extras
(need rework)

**Pretrained DALLE checkpoint**

[Download DALLE checkpoint](https://drive.google.com/file/d/111lGGkg0c7HPA5dBeLU8v7_pKWJWvuoT/view?usp=sharing)
 and place it in `ckpt/dalle/`
```
python bin/ensemble.py config/CLIP/dalle.yml \
    model.vae.image_size=256 \
    model.dalle.dim=1024 \
    model.dalle.heads=16 \
    model.dalle.dim_head=64 \
    model.dalle.depth=16 \
    model.vae.ckpt=data/CLIP/0.0022899999748915434_32_951498.pth \
    name="dalle" \
    ckpt_dir=ckpt \
    ckpt=1.661213994026184_4_406122.pth \
    ensemblor.generate_images=True
```

This trigger [the following code](https://github.com/jbdel/vilmedic/blob/main/vilmedic/networks/models/clip/DALLE.py#L17) 
that generates a few images for one sample.

## conVIRT

[Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://openreview.net/forum?id=T4gXBOXoIUr)

### Model
The model config is defined as such:
```
model:
  proto: ConVIRT
  encoder:
    proto: data/SELFSUP/huggingface/biomed_roberta_base
  cnn:
    proto: CNN
    backbone: resnet50
    output_layer: avgpool
    dropout_out: 0.0
    permute: batch_first
    freeze: False
  projection:
    visual_embedding_dim: 2048
    textual_embedding_dim: 768
    projection_dim: 768
  loss:
    proto: ConVIRTLoss
    tau: 0.1
    lambda_: 0.75
 ```
 

### Metrics and scores

| Dataset  |  batch-size |   Validation Loss  |  Config
| ------------- |:-------------:|:-------------:| :-------------:|
| **mimic-cxr**   | 
| [conVIRT](https://arxiv.org/pdf/2010.00747.pdf) (official splits)  | 32 | ~ 2.20
| ours (official splits) | 32  | 2.09 | [SELFSUP/convirt-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-mimic.yml)
| ours (balanced*)  | 32  | 1.65 | [SELFSUP/convirt-mimic-balanced.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-mimic-balanced.yml)
| **padchest**   | 
| ours (random splits**) | 16 | 2.26 | [SELFSUP/convirt-padchest.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-padchest.yml)
| ours (random splits**) | 32 | 2.91 | [SELFSUP/convirt-padchest.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-padchest.yml)
| **indiana**   | 
| ours (random splits**) | 16 | 1.61 | [SELFSUP/convirt-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-indiana.yml)
| ours (random splits**) | 32 | 1.97 | [SELFSUP/convirt-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-indiana.yml)
| ours (random splits**) | 64 | 2.59 | [SELFSUP/convirt-indiana.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/convirt-indiana.yml)

*\*balanced means redefining splits with an homogeneous distribution of the labels across the splits*<br/>
*\*\*No official splits exist*

### Extra

You can use the `plot_representation` post-process to plot learned representations:

```
ensemblor:
  batch_size: 32
  splits: [train, validate, test]
  post_processing:
    - plot_representation:
        keys:
          - linguistic
          - visual
        labels_keep: [Pleural Effusion,Pneumonia,Pneumothorax,Cardiomegaly,Atelectasis]
        max_samples_per_class: 250
  mode: best-1
```

Make sure to use a dataset that return labels:

``` 
python bin/ensemble.py config/SELFSUP/convirt-mimic.yml \
    dataset.proto=ImSeqLabel \
    dataset.label.root=data/SELFSUP/mimic-cxr/ \
    dataset.label.file=label.tok \
    ...
```

Here is the results on mimic-cxr (balanced):

| train full (linguistic)  |     train sampled (linguistic) | train full (visual) | train sampled (visual) 
| :-------------: |:-------------:|:-------------:|:-------------:|
[<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_linguistic.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_linguistic.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_linguistic.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_linguistic.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_visual.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_visual.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_visual.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_visual.png)
*Click image to access full-size*


## simCLR

[A Simple Framework for Contrastive Learning of Visual Representations](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)

### Model
The model config is defined as such:

``` 
model:
  proto: SimCLR
  cnn:
    proto: CNN
    backbone: resnet50
    output_layer: avgpool
    dropout_out: 0.0
    permute: batch_first
    freeze: False

  projection:
    visual_embedding_dim: 2048
    projection_dim: 768

  loss:
    tau: 0.5
```


### Metrics and scores

<div class="warning_box">
	<b>Warning: </b> When using <span class="div_pre">trainor.batch_size=16</span>, the batch-size 
	is actually of size 32 (16 images from the dataset + 16 corresponding enhanced images). See 
	the tranforms in <span class="div_pre">simclr-mimic.yml</span>.
</div>

| Dataset  |  batch-size |    Validation Loss | Config | 
| ------------- |:-------------:|:-------------:|:-------------:|
| **mimic-cxr**   | 
| ours (official splits)  | 32  | 1.96 | [SELFSUP/simclr-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/simclr-mimic.yml)
| ours (official splits)  | 64  | 2.48 | [SELFSUP/simclr-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/simclr-mimic.yml)
| ours (official splits)  | 128  | 3.06 | [SELFSUP/simclr-mimic.yml](https://github.com/jbdel/vilmedic/blob/main/config/SELFSUP/simclr-mimic.yml)

### Extra

You can use the `plot_representation` post-process to plot learned representations:

``` 
python bin/ensemble.py config/SELFSUP/simclr-mimic-eval.yml \
    ckpt_dir=ckpt \
    ensemblor.splits=[train,validate,test] \
    name="simclr_32"
```


## GLoRIA

[GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-Efficient Medical Image Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html)

### Model

``` 
model:
  proto: GLoRIA

  encoder:
    proto: data/SELFSUP/huggingface/Bio_ClinicalBERT
    last_n_layers: 4

  cnn:
    proto: CNN
    backbone: resnet50
    output_layer: avgpool
    dropout_out: 0.0
    permute: batch_first
    freeze: False

  visual_embedder:
    interm_feature_dim: 1024
    feature_dim: 2048

  loss:
    local_loss_weight: 1.0
    global_loss_weight: 1.0
    temp1: 4.0
    temp2: 5.0
    temp3: 10.0
```

### Metrics and scores
| Dataset  |  batch-size |    Validation Loss | Config | 
| ------------- |:-------------:|:-------------:|:-------------:|
| **chexpert-validate**   | 
| [official](https://github.com/marshuang80/gloria)  | 48  | 9.67 | [original repo](https://github.com/marshuang80/gloria/blob/main/configs/chexpert_pretrain_config.yaml)
| **mimic-cxr-validate**   | 
| ours  | 48  | 9.37 | [SELFSUP/gloria-mimic](https://github.com/jbdel/vilmedic/blob/main/config/RRG/biomed-roberta-baseline-mimic.yml)

