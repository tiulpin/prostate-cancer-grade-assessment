# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import sys

sys.path.append('.')

from src.pl_module import CoolSystem


class DummySystem(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = self.get_net()

    def forward(self, x: torch.tensor):
        return self.net(x)

    def get_net(self) -> torch.nn.Module:
        return CoolSystem.net_mapping(self.hparams.net, 5)


def main(checkpoint_path: str):
    pth_path = checkpoint_path[:len(checkpoint_path) - 4] + 'pth'
    model = DummySystem.load_from_checkpoint(checkpoint_path, )
    state_dict = model.net.state_dict()
    torch.save(state_dict, pth_path)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    main(args.checkpoint)
