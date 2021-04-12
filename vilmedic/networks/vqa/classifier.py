import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout, **kwargs):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_classes)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output: torch.Tensor, label: torch.Tensor):
        output = self.dropout(output)
        output = self.classifier(output)
        loss = self.loss_func(output, label)
        return {'loss': loss, 'output': output}
