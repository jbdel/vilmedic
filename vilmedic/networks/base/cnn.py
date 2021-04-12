import torch
import torch.nn as nn
from torchvision.models import *


class Network(nn.Module):
    def __init__(self, backbone, output_layer, dropout_out):
        super(Network, self).__init__()
        self.output_layer = output_layer
        self.backbone = backbone
        self.network = eval(backbone)(pretrained=True)
        self.dropout_out = nn.Dropout(p=dropout_out)

        layers = [n for n, _ in self.network.named_children()]
        assert output_layer in layers, '{} not in {}'.format(output_layer, layers)

        sub_network = []
        for n, c in self.network.named_children():
            sub_network.append(c)
            if n == output_layer:
                break

        self.network = nn.Sequential(*sub_network)

    def forward(self, images):
        out = self.network(images)
        out = self.dropout_out(out)
        return out

    def __repr__(self):
        s = str(self.backbone) + '(output_layer=' + self.output_layer + ',dropout_out=' + str(self.dropout_out.p) + ')'
        return s


class CNN(nn.Module):
    def __init__(self, backbone, dropout_out, output_layer, ctx_size, **kwargs):
        super(CNN, self).__init__()
        self.ctx_size = ctx_size
        self.network = Network(backbone, output_layer, dropout_out)

    def forward(self, images, **kwargs):
        mask = None
        out = self.network(images)
        out = out.view(*out.size()[:2], -1).permute(2, 0, 1)
        return out, mask
