<div class="warning_box">
	<b>Warning: </b> The models are resource-hungry. If you can't run a configuration because the training batch-size 
	is too big, you can use gradient accumulation as such:
	<div class="highlight">
<pre>python bin/train.py config/task/conf.yml \
    trainor.batch_size=8 \
    trainor.grad_accu=8     </pre></div>	
</div>

# Solutions

The following is a list of replicated solutions available in ViLMedic.

## Radiology Report Generation

### BioMed-RoBERTa baseline 
```bash
for i in {1..6}
do
    python bin/train.py config/RRG/biomed-roberta-baseline.yml \
        trainor.batch_size=64 \
        validator.batch_size=4 \
        name=rrg 
done
```

The model is defined as such in the config file:
```
model:
  proto: RRG
  decoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
  cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features
    dropout_out: 0.0
    permute: batch_first
    visual_embedding_dim: 1664
    freeze: False
```

### Results

```
python bin/ensemble.py config/RRG/biomed-roberta-baseline.yml \
    ensemblor.batch_size=4 \
    ensemblor.beam_width=8 \
    ensemblor.mode=best-1 \
    name=rrg 
```
| Dataset |     ROUGE-L | F1-cheXbert (micro) |  
| ------------- |:-------------:|:-------------:|
| **Mimic-test**
| [M2-Trans (2021)](https://arxiv.org/pdf/2010.10042.pdf) |  -  |  44.70 |
| BioMed-RoBERTa   | 22.46  |  45.04  |     


## Radiology Report Summarization
### Monomodal
```
for i in {1..6}
do
    python bin/train.py config/summarization/biorobert_mono.yml \
        trainor.batch_size=64 \
        validator.batch_size=4 \
        name=sum_mono
done
```
The model is defined as such in the config file:
```
model:
  proto: SumHugMono
  encoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
  decoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
```
#### Results
One model: 

```
python bin/ensemble.py config/RRG/biorobert_mono.yml \
    ensemblor.batch_size=4 \
    ensemblor.beam_width=8 \
    ensemblor.mode=best-1 \
    name=sum_mono 
```

| Split  |     BLEU | ROUGE2 | METEOR | ROUGEL
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Mimic-val   | 27.55   |  48.10  | 25.27    | 57.16 
| Mimic-test    |  19.58  |  33.63  | 21.02 | 45.16

## Medical VQA

```
for i in {1..10}
do
    python bin/train.py config/VQA/vqa.yml \
        trainor.batch_size=32 \
        validator.batch_size=4 \
        name=vqa
done
```
The model is defined as such in the config file:
```
model:
  proto: VQA_tr
  cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features
    dropout_out: 0.0
    permute: batch_first
    freeze: False

  adapter:
    input_size: 1664
    output_size: 768

  transformer:
    hidden_size: 768
    intermediate_size: 2048
    num_hidden_layers: 12
    num_attention_heads: 8
    attention_probs_dropout_prob: 0.1
    hidden_dropout_prob: 0.1
    hidden_act: gelu
    initializer_range: 0.02
    layer_norm_eps: 1.e-12

  classifier:
    proto: Classifier
    input_size: 768
    num_classes: 330
    dropout: 0.

  loss:
      proto: LabelSmoothingCrossEntropy
```
### Results

```
python bin/ensemble.py config/VQA/vqa_tr.yml \
    ensemblor.batch_size=4 \
    ensemblor.mode=best-7 \
    name=vqa
```

| Split  |  Model |   Accuracy | 
| ------------- |:-------------:|:-------------:|
| VQA-Med-2021 val-1 (in-domain)  | ours (single model) | 69.0
|   | ours (ens-7) | 72.0
|   | [SYSU-HCP (ens-8)](http://ceur-ws.org/Vol-2936/paper-99.pdf) | 69.2
| VQA-Med-2021 val-2 (out-domain)  | ours (ens-7)  | 36.1
|   | [SYSU-HCP (ens-8)](http://ceur-ws.org/Vol-2936/paper-99.pdf) | 38.2


## DALLE

<div class="data_box">
	<b>Data requirements: </b> mimic-cxr-images and CLIP
	<div class="highlight">
<pre>python data/download.py mimic-cxr-images-512,CLIP </pre></div>	
</div>


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

For details on the training parameters, please refer to the config files.

| Split  |     Loss | 
| ------------- |:-------------:|
| Mimic-CXR val   | 0.00345

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

| Split  |     Loss | 
| ------------- |:-------------:|
| Mimic-CXR val   | 1.6828  

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

<div class="data_box">
	<b>Data requirements: </b> mimic-cxr-images and SELFSUP data
	<div class="highlight">
<pre>python data/download.py mimic-cxr-images-512,SELFSUP </pre></div>	
</div>

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

You can train a model using the following command:

```
python bin/train.py config/SELFSUP/convirt-mimic.yml \
    trainor.batch_size=32 \
    trainor.optim_params.lr=5e-5 \
    trainor.optimizer=RAdam \
    trainor.lr_decay_params.patience=5 \
    trainor.early_stop=20 \
    model.forward_batch_size=4 \
    validator.batch_size=32
```   

Metrics and scores:

| Dataset  |     Validation Loss | 
| ------------- |:-------------:|
| **mimic-cxr**   | 
| [conVIRT](https://arxiv.org/pdf/2010.00747.pdf) (official splits)   | ~ 2.20
| ours (official splits)   | 2.09
| ours (balanced*)   | 1.65

*\*balanced means redefining splits with an homogeneous distribution of the labels across the splits*

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

Here is the results on mimic-cxr (balanced):

| train full (linguistic)  |     train sampled (linguistic) | train full (visual) | train sampled (visual) 
| :-------------: |:-------------:|:-------------:|:-------------:|
[<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_linguistic.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_linguistic.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_linguistic.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_linguistic.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_visual.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_full_visual.png) | [<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_visual.png" />](https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/convirt_train_sampled_visual.png)
*Click image to access full-size*
