import torch
import torch.nn as nn
import torchvision


class vgg16HGap(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg16HGap, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:16])
        self.conv4 = nn.Sequential(*self.model_list[16:23])
        self.conv5 = nn.Sequential(*self.model_list[23:30])
        self.conv6 = nn.Sequential(self.model_list[30])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        self.pool6 = nn.AdaptiveAvgPool2d(1)

        # self.lsoftmax_linear = LSoftmaxLinear(
        #     input_features=1984, output_features=num_classes, margin=margin, device=rank)

        # self.asoftmax = AngleLinear(1984, num_classes, m=margin)

    def forward(self, x, targeta=None, targetb=None):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)

        y2 = self.conv2(y1)
        p2 = self.pool2(y2)

        y3 = self.conv3(y2)
        p3 = self.pool3(y3)

        y4 = self.conv4(y3)
        p4 = self.pool4(y4)

        y5 = self.conv5(y4)
        p5 = self.pool5(y5)

        y6 = self.conv6(y5)
        p6 = self.pool6(y6)

        f = torch.cat([p1, p2, p3, p4, p5, p6], 1).squeeze(3).squeeze(2)

        return f

        # if self.training:
        #     outa = self.lsoftmax_linear(input=f, target=targeta)
        #     outb = self.lsoftmax_linear(input=f, target=targetb)
        #     return outa, outb
        # else:
        #     return self.lsoftmax_linear(input=f)

        # return self.asoftmax(f)


class vgg19HGap(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg19HGap, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=pretrained).features
        self.model_list = list(self.vgg.children())
        self.conv1 = nn.Sequential(*self.model_list[0:4])
        self.conv2 = nn.Sequential(*self.model_list[4:9])
        self.conv3 = nn.Sequential(*self.model_list[9:18])
        self.conv4 = nn.Sequential(*self.model_list[18:27])
        self.conv5 = nn.Sequential(*self.model_list[27:36])
        self.conv6 = nn.Sequential(self.model_list[36])
        self.pool1 = nn.AvgPool2d(224)
        self.pool2 = nn.AvgPool2d(112)
        self.pool3 = nn.AvgPool2d(56)
        self.pool4 = nn.AvgPool2d(28)
        self.pool5 = nn.AvgPool2d(14)
        self.pool6 = nn.AvgPool2d(7)


    def forward(self, x):
        y1 = self.conv1(x)
        p1 = self.pool1(y1)

        y2 = self.conv2(y1)
        p2 = self.pool2(y2)

        y3 = self.conv3(y2)
        p3 = self.pool3(y3)

        y4 = self.conv4(y3)
        p4 = self.pool4(y4)

        y5 = self.conv5(y4)
        p5 = self.pool5(y5)

        y6 = self.conv6(y5)
        p6 = self.pool6(y6)

        out = torch.cat([p1, p2, p3, p4, p5, p6], 1).squeeze(3)
        out = out.squeeze(2)
        return out
