import torch
import torch.nn as nn
from torchvision.models import *


class Network(nn.Module):
    def __init__(self, backbone, output_layer, dropout_out, freeze, **kwargs):
        super(Network, self).__init__()
        self.output_layer = output_layer
        self.backbone = backbone

        self.network = eval(backbone)(pretrained=True)
        self.dropout_out = nn.Dropout(p=dropout_out)
        self.freeze = freeze

        layers = [n for n, _ in self.network.named_children()]
        assert output_layer in layers, '{} not in {}'.format(output_layer, layers)

        sub_network = []
        for n, c in self.network.named_children():
            sub_network.append(c)
            if n == output_layer:
                break

        self.network = nn.Sequential(*sub_network)
        if freeze:
            for name, param in self.network.named_parameters():
                param.requires_grad = False

    def forward(self, images):
        out = self.network(images)
        out = self.dropout_out(out)
        return out


class CNN(nn.Module):
    def __init__(self, backbone, dropout_out, output_layer, permute, freeze=True, **kwargs):
        super(CNN, self).__init__()
        self.network = Network(backbone, output_layer, dropout_out, freeze, **kwargs)
        self.permute = permute
        self.freeze = freeze
        assert permute in ["batch_first", "spatial_first", "no_permute"]

    def forward(self, images, **kwargs):
        mask = None
        out = self.network(images)

        if self.permute == "no_permute":
            out = out
        elif self.permute == "batch_first":
            out = out.view(*out.size()[:2], -1).permute(0, 2, 1)
        elif self.permute == "spatial_first":
            out = out.view(*out.size()[:2], -1).permute(2, 0, 1)
        else:
            raise NotImplementedError()

        return out, mask

    def train(self, mode: bool = True):
        if self.freeze:
            mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def __repr__(self):
        s = str(self.network.backbone) + '(output_layer=' + self.network.output_layer + ', dropout_out=' + str(
            self.network.dropout_out.p) + ', freeze=' + str(self.freeze) + ')'
        return s
