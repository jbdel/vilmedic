import torch
import torch.nn as nn


class ConVIRTLoss(nn.Module):
    def __init__(self, tau, lambda_, **kwargs):
        super().__init__()
        self.tau = tau
        self.lambda_ = lambda_
        self.cos_loss = nn.CosineSimilarity()

    def forward(self, linguistic, visual):
        nominator = torch.exp(torch.div(self.cos_loss(linguistic, visual), self.tau))

        denominator_l = self.pairwise_cosine_distance(linguistic, visual)
        denominator_l = torch.sum(torch.exp(torch.div(denominator_l, self.tau)), dim=1)
        loss_l = -torch.log(torch.div(nominator, denominator_l))

        denominator_v = self.pairwise_cosine_distance(visual, linguistic)
        denominator_v = torch.sum(torch.exp(torch.div(denominator_v, self.tau)), dim=1)
        loss_v = -torch.log(torch.div(nominator, denominator_v))
        loss = torch.mean(self.lambda_ * loss_v + (1 - self.lambda_) * loss_l)
        return loss, loss_l, loss_v

    @staticmethod
    def pairwise_cosine_distance(a, b):
        eps = 1e-08
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def __repr__(self):
        return "ConVIRTLoss(\n" + \
               "\t(cos_loss): CosineSimilarity()\n" + \
               "\t(tau): {}\n".format(self.tau) + \
               "\t(lambda_): {}\n".format(self.lambda_) + \
               ")"
