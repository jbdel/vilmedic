import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, tau, **kwargs):
        super().__init__()
        self.tau = tau
        self.xe = nn.CrossEntropyLoss(reduction='none')

    def forward(self, linguistic, visual):
        n = linguistic.shape[0]
        logits = linguistic @ visual.T
        labels = torch.arange(n).cuda()
        loss_t = self.xe(logits, labels)
        loss_i = self.xe(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        loss = loss.mean()
        return loss, loss_t, loss_i

    def __repr__(self):
        return "InfoNCELoss(\n" + \
               "\t(tau): {}\n".format(self.tau) + \
               ")"
