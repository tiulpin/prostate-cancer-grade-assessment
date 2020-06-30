import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self,):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=128)
        self.conv2 = nn.Conv3d(
            in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1
        )
        self.bn2 = nn.BatchNorm3d(num_features=128)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)

        return out
