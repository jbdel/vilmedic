from torch import Tensor
from torch.nn import Sequential, Linear, Dropout, Module

class Classifier(Module):
    def __init__(self, input_size, num_classes, dropout=0., **kwargs):
        super(Classifier, self).__init__()
        self.classifier = Sequential(
            Linear(in_features=input_size, out_features=num_classes)
        )
        self.dropout = Dropout(p=dropout)

    def forward(self, input: Tensor):
        output = self.dropout(input)
        output = self.classifier(output)
        return output
