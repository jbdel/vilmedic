# Overview

A block is a snippet of code, usually written in
PyTorch, that contains a sub-part of a solution. It
can be a piece of a neural network architecture, a
loss function, or an evaluation metric. Therefore, a
block can be suitable for several solutions.

## Using blocks

You can define a block in a configuration file as such:
```
my_cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features
```
In this example, we are instantiating a `CNN` block with name `my_cnn`. Blocks declaration must respect their usage as stated in the documentation.

You can pass blocks to the `my_model` class using the `model` key of the configuration file.
```
model:
  proto: my_model
  my_cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features

  transformer:
    proto: DecoderModel
    hidden_size: 768
    intermediate_size: 2048
    num_hidden_layers: 12
```
And catch it in `my_model` as such:

```
class my_model(nn.Module):
    def __init__(self, my_cnn, transformer, **kwargs):
        super().__init__()
        # Encoder
        self.my_cnn = eval(my_cnn.pop('proto'))(**my_cnn)
        # Decoder
        self.dec = eval(cnn.pop('transformer'))(**transformer)
    
    # Build your solution based on these bocks.
```