import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def CELossWithWeight(weight):
    r"""
    weight = torch.FloatTensor([50000, 10000]) # [negative sample, positive sample]
    """
    weight = torch.max(weight) / weight
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    return criterion

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss