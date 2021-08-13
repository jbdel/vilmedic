import torch
import torch.nn as nn
from vilmedic.networks.vision import *
from vilmedic.networks.classifier import *
from vilmedic.networks.classifier.evaluation import evaluation
from vilmedic.networks.classifier.losses import get_loss

from .utils import get_n_params


class VQA(nn.Module):
    def __init__(self, visual, classif, loss, **kwargs):
        super(VQA, self).__init__()

        visual_func = visual.pop('proto')
        # classif_func = classif.pop('proto')
        loss_func = loss.pop('proto')

        self.encoder = eval(visual_func)(**visual)
        # self.classifier = eval(classif_func)(**classif)
        self.encoder.cnn.classifier = nn.Sequential(
            nn.Linear(classif.pop('input_size'), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classif.pop('num_classes')),
        )
        self.loss_func = get_loss(loss_func, **loss).cuda()
        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, labels, **kwargs):
        output, mask = self.encoder(images.cuda())
        loss = self.loss_func(output, labels.cuda(), **kwargs)
        return {'loss': loss, 'output': output}

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
