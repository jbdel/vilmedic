import torch
import torch.nn as nn
from torchvision.models import *
from torchxrayvision.models import DenseNet as XrvDenseNet, ResNet as XrvResNet
from .vgg_hgap import *


def get_network(backbone, output_layer, pretrained, weights=None, **kwargs):
    """
    Create sub-network given a backbone and an output_layer
    """
    # Create avgpool for densenet, doesnt exist as such
    if 'densenet' in backbone and output_layer == 'avgpool':
        sub_network = get_network(backbone, 'features', pretrained, **kwargs)
        sub_network.add_module('relu', nn.ReLU(inplace=True))
        sub_network.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        sub_network.add_module('flatten', nn.Flatten(1))
        return sub_network

    # torchxrayvision
    if "xrv" in backbone.lower():
        network = eval(backbone)(weights=weights)
        if hasattr(network, 'model'):  # XrvResNet Fix
            network = network.model
    else:
        network = eval(backbone)(pretrained=pretrained, **kwargs)

    if output_layer is not None and (not output_layer == 'classifier'):
        layers = [n for n, _ in network.named_children()]
        assert output_layer in layers, '{} not in {}'.format(output_layer, layers)
        sub_network = []
        for n, c in network.named_children():
            sub_network.append(c)
            if n == output_layer:
                break
        network = nn.Sequential(*sub_network)

    return network


class CNN(nn.Module):
    def __init__(self, backbone, dropout_out, permute, freeze=True, output_layer=None, pretrained=True,
                 visual_embedding_dim=None, **kwargs):
        super(CNN, self).__init__()
        self.backbone = backbone
        self.output_layer = output_layer
        self.permute = permute
        self.freeze = freeze
        self.pretrained = pretrained

        self.cnn = get_network(self.backbone, self.output_layer, self.pretrained, **kwargs)
        self.dropout_out = nn.Dropout(p=dropout_out)

        assert permute in ["batch_first", "spatial_first", "no_permute"]

        if freeze:
            for name, param in self.cnn.named_parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs):
        out = self.cnn(images)
        out = self.dropout_out(out)
        if self.permute == "no_permute":
            out = out
        elif self.permute == "batch_first":
            out = out.view(*out.size()[:2], -1).permute(0, 2, 1)
            if out.shape[1] == 1:  # avgpool case
                out = out.squeeze(1)
        elif self.permute == "spatial_first":
            out = out.view(*out.size()[:2], -1).permute(2, 0, 1)
        else:
            raise NotImplementedError()

        return out

    def train(self, mode: bool = True):
        if self.freeze:
            mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def __repr__(self):
        s = str(self.backbone) + '(output_layer=' + self.output_layer + ', dropout_out=' + str(
            self.dropout_out.p) + ', freeze=' + str(self.freeze) + ', pretrained=' + str(self.pretrained) + \
            ('\n classifier= {}'.format(self.cnn.classifier) if self.output_layer == 'classifier' else '' + ')')
        return s
