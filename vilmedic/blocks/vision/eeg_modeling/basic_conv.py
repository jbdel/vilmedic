import torch.nn as nn
import torch.nn.functional as F

"""
Baisic conv2d modules, can specify whether to use batchnorm.
Future: Specify nonlinear activation
"""


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return F.relu(x, inplace=True)
