# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

import timm
import torch
import torch.nn as nn


class EffNetRegressor(nn.Module):

    def __init__(self, backbone: str, out_dim: int):
        super(EffNetRegressor, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.fc = nn.Linear(self.backbone.classifier.in_features, out_dim)
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = EffNetRegressor('efficientnet_b0', 5)
    print(model)
    x = torch.randn((1, 3, 500, 500))
    out = model(x)
    print(out.shape)
