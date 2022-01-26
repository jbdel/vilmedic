# Medical VQA

Since patients may now access structured and unstructured data related to their health via patient portals, such access also motivates the need to help them better understand their conditions regarding their available data, including medical images.

<div class="data_box">
	<b>Data requirements: </b> imageclef-vqa-images and MVQA data<br/>
	Please refer to <a href="https://vilmedic.readthedocs.io/en/latest/vilmedic/solutions/data.html">the data section</a>.
</div>


## Model
The model is defined as such in the config file:
```
model:
  proto: MVQA
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
## Metrics and scores

Dataset |   Accuracy | Config
| ------------- |:-------------:|-------------:|
|**VQA-Med-2021 val-1 (in-domain)** |
 | ours (single model) | 69.0 | [MVQA/vqa.yml](https://github.com/jbdel/vilmedic/blob/main/config/MVQA/vqa.yml)
   | ours (ens-7) | 72.0| [MVQA/vqa.yml](https://github.com/jbdel/vilmedic/blob/main/config/MVQA/vqa.yml)
   | [SYSU-HCP (ens-8)](http://ceur-ws.org/Vol-2936/paper-99.pdf) | 69.2
| **VQA-Med-2021 val-2 (out-domain)**
| ours (ens-7)  | 37.8| [MVQA/vqa.yml](https://github.com/jbdel/vilmedic/blob/main/config/MVQA/vqa.yml)
   | [SYSU-HCP (ens-8)](http://ceur-ws.org/Vol-2936/paper-99.pdf) | 38.2

## Extras
To ensemble, train 7 models and then do:

```
python bin/ensemble.py config/VQA/vqa_tr.yml \
    ensemblor.batch_size=4 \
    ensemblor.mode=best-7
```