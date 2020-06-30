# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import torch
from torch import nn


class WNAELoss(nn.Module):
    def __init__(self, w=None):
        super().__init__()
        if w is None:
            w = [0.3, 0.175, 0.175, 0.175, 0.175]
        self.w = torch.FloatTensor(w).cuda()

    def forward(self, output, target):
        return torch.sum(
            self.w
            * torch.sum(torch.abs(target - output), axis=0)
            / torch.sum(target, axis=0)
        )
