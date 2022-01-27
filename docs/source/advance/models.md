<div class="data_box">
	Models are defined in: 	<div class="highlight">
<pre>vilmedic/models</pre></div>
</div>

# Models

A model is a full solution in itself. It takes care of the inputs and outputs of that solution during training and 
validation. 
For example, a Radiology Report Generation model would output [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) during training and 
hypotheses and references during evaluation (to compute NLG metrics such as BLEU, ROUGE, ...). 

It usually consists of :
1. a neural network
1. a loss function
1. an evaluation method. 

Those three components can be defined by the user of vilmedic blocks. 

## Define a model

Create a python class within the folder `vilmedic/models` (or a new subfolder) that implements `nn.Module`.
```
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, image, input_ids, attention_mask):
        return {"loss": 0.0}
```
and declare your `MyModel` class in `networks/__init__.py`.

<div class="warning_box">
	By default, your model receives the pytorch training dataloader and the training logger.
<div class="highlight">
<pre>
print(kwargs)
>> {'dl': &lt;torch.utils.data.dataloader.DataLoader object at 0x7f26985b2eb0&gt;,
    'logger': &lt;Logger 406482 (SETTINGS)&gt;}
</pre></div>	
</div>

Finally, define the parameters your model should receive in a config file:
```
model:
  proto: MyModel
  linear:
    in_features: 512
    out_features: 1024
  use_relu: true
```
and catch it in your model as such:
```
class MyModel(nn.Module):
    def __init__(self, linear, use_relu, **kwargs):
        super().__init__()
        self.layer = nn.Linear(linear.in_features, linear.out_features)
        if use_relu:
            self.layer = nn.Sequential(self.layer, nn.ReLU())
        print(self)

    def forward(self, image, input_ids, attention_mask):
        return {"loss": 0.0}

>>> MyModel(
  (layer): Sequential(
    (0): Linear(in_features=512, out_features=1024, bias=True)
    (1): ReLU()
  )
)

```

Finally, build your forward function. During training your model should at least return a dictionary with the key "loss" 
and a tensor that can be used with a pytorch optimizer as value (typically the output of a pytorch loss function).

<div class="warning_box">
	If you want to take care of the optimization yourself, simply do not return a dictionary with "loss" as a key. Whatever you return will 
	then be printed on screen as a log.
</div>

In this example, the forward method signature:
``` 
def forward(self, image, input_ids, attention_mask):
```
is written so that it works with the "[ImSeq](https://github.com/jbdel/vilmedic/blob/main/vilmedic/datasets/ImSeq.py#L30)" dataset collate function.


## Handling evaluation

Again, you are free to evaluate your model as you see fit. To do so, your model **must** have a ``eval_func`` attribute that stores 
your evaluation function. 

```
def my_evaluation(models, dl, **kwargs):
    print(kwargs)
    return {'loss': 0.0}
    

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eval_func = my_evaluation
 
    def forward(self, image, input_ids, attention_mask):
        return {"loss": 0.0}

>>> {'config': {'beam_size': 16,
              'metrics': ['BLEU', 'METEOR']
              },
    'from_training': true}
```
Note that your evaluation function will receive by default a list models, config (from config file, cf config section), dl (evaluation dataloader)
and `from_training` argument.

<div class="warning_box">
The <span class="div_pre">model</span> argument is a list of models in evaluation mode (i.e. <span class="div_pre">eval()</span>). 
If <span class="div_pre">from_training</span> is <span class="div_pre">True</span>, then the list will contain only one model, the one 
currently being trained. If <span class="div_pre">from_training</span> is <span class="div_pre">False</span> then it means that the Ensemblor 
called your evaluation function with one or several trained models. 

If your evaluation does not support model ensembling, then simply do:
<div class="highlight">
<pre>
def my_evaluation(models, dl, **kwargs):
    model = models[0]
</pre></div>	

</div>