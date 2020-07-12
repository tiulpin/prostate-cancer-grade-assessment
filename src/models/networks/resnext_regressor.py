# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

import timm
import torch
import torch.nn as nn


class ResNeXtRegressor(nn.Module):

    def __init__(self, backbone: str, out_dim: int):
        super(ResNeXtRegressor, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dense = nn.Linear(self.backbone.fc.in_features, out_dim)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    model = ResNeXtRegressor('resnext50_32x4d', 5)
    print(model)
    x = torch.randn((1, 3, 1536, 1536))
    out = model(x)
    print(out.shape)
