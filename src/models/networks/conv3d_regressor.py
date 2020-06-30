# coding: utf-8
__author__ = "bejeweled: https://kaggle.com/bejeweled"

import torch
from torch import nn
from torch.nn import functional as F

from src.models.layers.res_block import ResBlock


class Conv3DRegressor(nn.Module):
    def __init__(self, n_classes: int = 5):
        super(Conv3DRegressor, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=53, out_channels=128, kernel_size=(3, 3, 3), padding=2
        )
        self.bn1 = nn.BatchNorm3d(num_features=128)
        self.m_pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.res_conv1 = ResBlock()
        self.m_pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.res_conv2 = ResBlock()
        self.adap_pool = nn.AdaptiveMaxPool3d(output_size=(4, 4, 4))
        self.res_conv3 = ResBlock()
        self.fc1 = nn.Linear(in_features=128 * 4 * 4 * 4, out_features=256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=n_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.m_pool1(x)
        x = self.res_conv1(x)
        x = self.m_pool2(x)
        x = self.res_conv2(x)
        x = self.adap_pool(x)
        x = self.res_conv3(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()
