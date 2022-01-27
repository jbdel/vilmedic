import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0., **kwargs):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_classes)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor):
        output = self.dropout(input)
        output = self.classifier(output)
        return output
