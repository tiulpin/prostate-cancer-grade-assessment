# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

import timm
import torch
import torch.nn as nn

from src.models.layers.adaptive import AdaptiveConcatPool2d


class EffNetClassifier(nn.Module):
    def __init__(self, backbone: str, out_dim: int):
        super(EffNetClassifier, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.fc = nn.Linear(self.backbone.classifier.in_features, out_dim)
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        return x


class EffNetDoubleClassifier(nn.Module):
    def __init__(self, backbone: str, out_dim: int):
        super(EffNetDoubleClassifier, self).__init__()
        backbone = timm.create_model(backbone, pretrained=True)
        dimension = backbone.classifier.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            AdaptiveConcatPool2d((1, 1)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * dimension, dimension // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(dimension // 2, out_dim),
        )
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = EffNetDoubleClassifier("efficientnet_b0", 5)
    print(model)
    x = torch.randn((1, 3, 500, 500))
    out = model(x)
    print(out.shape)
