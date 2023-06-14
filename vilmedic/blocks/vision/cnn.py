import torch
import torch.nn as nn
import logging
from torchvision.models import *
from torchxrayvision.models import DenseNet as XrvDenseNet, ResNet as XrvResNet
from monai.networks.nets.densenet import densenet121 as MonaiDensenet121, densenet169 as MonaiDensenet169, \
    densenet201 as MonaiDensenet201, densenet264 as MonaiDensenet264

from .vgg_hgap import *
from transformers.models.vit.modeling_vit import ViTModel, ViTConfig
from transformers.models.deit.modeling_deit import DeiTModel, DeiTConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndNoAttention, \
    BaseModelOutputWithNoAttention
from transformers import ResNetConfig, PoolFormerConfig
from transformers.models.resnet.modeling_resnet import ResNetModel as HFResNetModel
from transformers.models.poolformer.modeling_poolformer import PoolFormerModel as HFPoolFormerModel


class _3d_densenet121(MonaiDensenet121):
    pass


class _3d_densenet169(MonaiDensenet169):
    pass


class _3d_densenet201(MonaiDensenet201):
    pass


class _3d_densenet264(MonaiDensenet264):
    pass


def get_network(backbone, output_layer, pretrained, weights=None, **kwargs):
    """
    Create sub-network given a backbone and an output_layer
    """
    # Create avgpool for densenet, does not exist as such
    if 'densenet' in backbone and '3d' not in backbone and output_layer == 'avgpool':
        sub_network = get_network(backbone, 'features', pretrained, **kwargs)
        sub_network.add_module('relu', nn.ReLU(inplace=True))
        sub_network.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        sub_network.add_module('flatten', nn.Flatten(1))
        return sub_network

    # HuggingFace Vision transformer
    if "vit" in backbone.lower():
        model = ViTModel(ViTConfig(return_dict=True, **kwargs), add_pooling_layer=False)
        model.layernorm = nn.Identity()
        return model

    if "deit" in backbone.lower():
        return DeiTModel(DeiTConfig(return_dict=True, **kwargs), add_pooling_layer=False)

    # HuggingFace ResNet
    if "hfresnet" in backbone.lower():
        return HFResNetModel(ResNetConfig(return_dict=True, **kwargs))

    # HuggingFace PoolFormer
    if "hfpoolformer" in backbone.lower():
        return HFPoolFormerModel(PoolFormerConfig(return_dict=True, **kwargs))

    # Torchxrayvision
    if "xrv" in backbone.lower():
        network = eval(backbone)(weights=weights)
        if hasattr(network, 'model'):  # XrvResNet Fix
            network = network.model
    # PyTorch
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
    def __init__(self,
                 backbone,
                 permute,
                 dropout_out=0.0,
                 freeze=False,
                 output_layer=None,
                 pretrained=True,
                 visual_embedding_dim=None,
                 **kwargs):
        super(CNN, self).__init__()

        self.backbone = backbone
        self.output_layer = output_layer
        self.permute = permute
        self.freeze = freeze
        self.pretrained = pretrained
        self.cnn = get_network(self.backbone, self.output_layer, self.pretrained, **kwargs)
        self.dropout_out = nn.Dropout(p=dropout_out)
        self.is3D = "3d" in backbone

        assert permute in ["batch_first", "spatial_first", "no_permute"]

        if freeze:
            for name, param in self.cnn.named_parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs):
        out = self.cnn(images)
        if isinstance(self.cnn, ViTModel) or isinstance(self.cnn, DeiTModel):
            assert isinstance(out, BaseModelOutputWithPooling)
            out = self.dropout_out(out.last_hidden_state)
            return out

        if isinstance(self.cnn, HFResNetModel):
            assert isinstance(out, BaseModelOutputWithPoolingAndNoAttention)
            out = out.last_hidden_state

        if isinstance(self.cnn, HFPoolFormerModel):
            assert isinstance(out, BaseModelOutputWithNoAttention)
            out = out.last_hidden_state

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
        s = str(self.backbone) + \
            '(' + \
            (str(type(self.cnn).__name__) + '(' + str(self.cnn.config) + '), ' if self.backbone.lower() in ['deit',
                                                                                                            'vit'] else '') + \
            'dropout_out=' + str(self.dropout_out.p) + \
            ', freeze=' + str(self.freeze) + \
            (', output_layer=' + str(self.output_layer) if self.output_layer is not None else '') + \
            (', pretrained=' + str(self.pretrained) if self.backbone.lower() not in ['deit', 'vit'] else '') + \
            ('\n classifier= {}'.format(list(self.cnn.children())[-1]) if self.output_layer == 'classifier' else '' + ')')
        return s
