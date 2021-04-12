import torch
import torch.nn as nn
import numpy as np
from .cnn import CNN
from vilmedic.networks.rnn.textencoder import TextEncoder
from vilmedic.networks.rnn.utils import get_n_params
from .classifier import Classifier
from .evaluation import evaluation

class VQA(nn.Module):
    def __init__(self, visual, linguistic, classif, feats_rnn=False, **kwargs):
        super(VQA, self).__init__()

        visual_func = visual.proto
        linguistic_func = linguistic.proto
        classif_func = classif.proto

        self.cnn = eval(visual_func)(**vars(visual))
        self.rnn = eval(linguistic_func)(**vars(linguistic))
        self.classifier = eval(classif_func)(**vars(classif))
        self.feats_rnn = feats_rnn

        # Evaluation
        self.eval_func = evaluation


    def encode(self, question, image, **kwargs):
        feats, mask = self.cnn(image.cuda())
        if self.feats_rnn:
            return {'question': self.rnn(question, feats), 'image': (feats, mask)}
        else:
            return {'question': self.rnn(question, None), 'image': (feats, mask)}

    def forward(self, image, question, label, **kwargs):
        enc = self.encode(question.cuda(), image.cuda())

        hidden_states, _ = enc['question']
        last_hidden_state = hidden_states[:, -1, :]
        feats, _ = enc['image']
        output = torch.cat((last_hidden_state, feats), -1)
        result = self.classifier(output, label.cuda())

        return {'loss': result['loss'], 'output': result['output'], 'label': label}

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
