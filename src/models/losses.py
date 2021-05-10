from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
"""
Entropy Loss adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf
"""

# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()

#     def forward(self, x):
#         # b = F.sigmoid(x) * F.logsigmoid(x)
#         # b = -1.0 * b.sum().mean()

#         b_0 = torch.sigmoid(x)
#         b_1 = F.logsigmoid(x)
#         b = b_0 * b_1
#         b = -1.0 * b.sum().mean()
#         return b
#         # return -torch.sum(torch.mul(F.sigmoid(x), torch.log2(F.sigmoid(x)))).mean()


class HLoss(nn.Module):
    def __init__(self):
        self.sigmoid = torch.nn.Sigmoid()
        self.log_sigmoid = torch.nn.LogSigmoid()
        super(HLoss, self).__init__()

    def forward(self, x):
        b = self.sigmoid(x) * self.log_sigmoid(x)
        b = -1.0 * b.sum(1)
        return b.mean()


class DSCLoss(nn.Module):
    def __init__(self, dsc_loss_coeff):
        super(DSCLoss, self).__init__()

        self.gamma = dsc_loss_coeff
        self.base = torch.nn.BCEWithLogitsLoss()

    def __call__(self, scores, y):

        base = self.base(scores, y)

        if self.gamma > 0:
            p = scores.sigmoid()
            p = p.view(-1, 1)
            y = y.view(1, -1)
            numer = 2 * y @ p
            denom = y.sum() + p.sum()
            dsc_loss = numer / denom
        else:
            dsc_loss = 0

        return base - dsc_loss


class DSCLoss_2(nn.Module):
    def __init__(self):
        super(DSCLoss_2).__init__()

    def __call__(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))
