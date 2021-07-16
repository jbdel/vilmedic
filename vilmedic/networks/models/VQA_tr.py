import torch
import torch.nn as nn
from vilmedic.networks.vision import *
from vilmedic.networks.classifier import *
from vilmedic.networks.rnn.utils import get_n_params
from vilmedic.networks.classifier.evaluation import evaluation
from vilmedic.networks.classifier.losses import get_loss
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.models.bert_generation import BertGenerationConfig

class VQA_tr(nn.Module):
    def __init__(self, visual, classif, adapter, transformer, loss, **kwargs):
        super(VQA_tr, self).__init__()

        visual_func = visual.pop('proto')
        loss_func = loss.pop('proto')
        bert_conf = BertGenerationConfig(**transformer)

        self.encoder = eval(visual_func)(**visual)
        self.adapter = nn.Sequential(
            nn.Linear(adapter.pop('input_size'), adapter.pop('output_size')),
            torch.nn.LayerNorm(bert_conf.hidden_size, eps=bert_conf.layer_norm_eps)
        )

        self.transformer = BertEncoder(bert_conf)
        self.pooler = BertPooler(bert_conf)
        # self.classifier = nn.Sequential(
        #     nn.Linear(classif.pop('input_size'), 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, classif.pop('num_classes')),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(classif.pop('input_size'), classif.pop('num_classes')),
        )

        self.loss_func = get_loss(loss_func, **loss).cuda()
        # Evaluation
        self.eval_func = evaluation

    def forward(self, image, label, **kwargs):
        out, mask = self.encoder(image.cuda())
        out = self.adapter(out)
        out = self.transformer(out)
        out = out[0]
        out = self.pooler(out)
        out = self.classifier(out)
        loss = self.loss_func(out, label.cuda(), **kwargs)
        return {'loss': loss, 'output': out}

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
