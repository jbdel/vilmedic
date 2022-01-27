# Configuration files

The minimal configuration file looks like this:

```
name: my_experiment
ckpt_dir: ckpt
dataset:
  proto: my_dataset
model:
  proto: my_model
trainor:
  optimizer: RAdam
  batch_size: 64
validator:
  batch_size: 16
  metrics: [accuracy]
  splits: [validate]
ensemblor:
  batch_size: 16
  metrics: [accuracy, f1-score, auroc]
  splits: [validate, test]
```

#### Name and exp <br/>
<hr/>

All output files (logs, checkpoints,... ) will be stored in `ckpt/my_exp`.

<br/>

#### Dataset <br/>
<hr/>

`proto` contains the dataset classname existing in the [dataset folder](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/__init__.py).
<br/>

You can pass arguments to the `my_dataset` class as such:

```
dataset:
  proto: my_dataset
  image:
    root: data/rrg/mimic-cxr/
    file: image.tok
    image_path: data/report_sum/
    load_memory: False
    ext: .jpg
```

And catch it in the dataset as such:

```
class my_dataset(Dataset):
    def __init__(self, image, **kwargs):
        super().__init__()
        self.image_path = image["image_path"]    
```



<br/>

#### Model <br/>
<hr/>

`proto` contains the model classname existing in the [model folder](https://github.com/jbdel/vilmedic/blob/main/vilmedic/models/__init__.py).
<br/>
You can pass blocks to the `my_model` class as such:

```
model:
  proto: my_model
  cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features

  transformer:
    proto: HFDecoder
    hidden_size: 768
    intermediate_size: 2048
    num_hidden_layers: 12
```

And catch it in the model as such:

```
class my_model(nn.Module):
    def __init__(self, cnn, transformer, **kwargs):
        super().__init__()
        # Encoder
        self.cnn = eval(cnn.pop('proto'))(**cnn)
        # Decoder
        self.dec = eval(cnn.pop('transformer'))(**transformer)
    
    # Build you solution base on these bocks.
```

<br/>

#### Trainor <br/>
<hr/>

Defines the training loop. Here are options available

```
trainor:
  batch_size: 64
  optimizer: RAdam
  optim_params: {lr: 0.0005, weight_decay: 0.00001}
  lr_decay_factor: 0.5
  lr_decay_patience: 1
  lr_min: 0.000001
  epochs: 99
  eval_start: 0
  early_stop: 10
  early_stop_metric: BLEU
```
- optimizer: can be a class from [torch.optim](https://pytorch.org/docs/stable/optim.html) or [pytorch_optimizer](https://github.com/jettify/pytorch-optimizer). 
`optim_params` are passed to the optimizer class. 
- early_stop: defines how many evaluation should be carried out without improvement on `early_stop_metric` metric before stopping training

<br/>

#### Validator <br/>
<hr/>


Defines the validation loop. Here are options available:
``` 
validator:
  batch_size: 16
  beam_width: 8
  metrics: [ROUGE2, BLEU, METEOR]
  splits: [validate]
```
- beam_width: can be specified for language generation tasks
- metrics: used during evaluation
- splits: specifies the filename pattern used by the Dataset class during evaluation (i.e. dataloader will get all the 
validate* files in specified data folder, more infos in the dataset section). If multiple splits are defined, the evaluator 
will return metrics for each split and the Trainor will use the averaged `early_stop_metric` across the splits.

<br/>

#### Ensemblor <br/>
<hr/>


Defines the ensemble script. Here are options available:
```
ensemblor:
  batch_size: 16
  beam_width: 8
  metrics: [ROUGE2, BLEU, METEOR]
  splits: [validate, test]
  mode: all
```
- beam_width: can be specified for language generation tasks
- mode: specifies how many checkpoint should be ensemble. If `all`, the Ensemblor considers all checkpoints in `ckpt/my_exp`. If `best-n`,
 the Ensemblor takes the `n` best checkpoints according to `early_stop_metric`. If mode is a path, i.e. `ckpt/my_exp/50.36_10_0.ckpt`, only evaluate this checkpoint.
