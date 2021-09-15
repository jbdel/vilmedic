<div style="warning_box">
	<b>Warning: </b> The models are resource-hungry. If you can run a configuration because the training batch-size 
	is too big, you can use the following option:
<pre>python bin/train.py config/RRG/rrg.yml \
    trainor.batch_size=8 \
    trainor.grad_accu=8     </pre>	
</div>

# Models

Here is the list of all available models in ViLMedic

## Radiology Report Generation

```bash
for i in {1..6}
do
    python bin/train.py config/RRG/rrg.yml \
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
python bin/ensemble.py config/RRG/rrg.yml \
    ensemblor.batch_size=4 \
    ensemblor.beam_width=8 \
    ensemblor.mode=best-1 \
    name=rrg 
```
| Split  |     BLEU | ROUGE2 | METEOR | CiDER | 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Mimic-val   | 43.61   |  43.90  | 45.36  | 6      
| Mimic-test    |  82.30  |  81.53  | 82.26  |  8        

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
python bin/ensemble.py config/CLIP/dalle.yml \
        model.vae.image_size=256 \
        model.dalle.dim=768 \
        model.dalle.heads=12 \
        model.dalle.dim_head=64 \
        model.dalle.depth=8 \
        model.vae.ckpt=ckpt/vae.pth \
        trainor.batch_size=8 \
        trainor.optim_params.lr=3e-4 \
        trainor.grad_accu=8 \
        name=dalle
```     

| Split  |     Loss | 
| ------------- |:-------------:|
| Mimic-CXR val   | 1.6828  
