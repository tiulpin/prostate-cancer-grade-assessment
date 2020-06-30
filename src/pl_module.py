# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"


import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.datasets.trends_np import TrendsNpSet
from src.losses.w_nae import WNAELoss
from src.models.networks.conv3d_regressor import Conv3DRegressor


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return {
            "loss": loss,
            "log": {f"train/{self.hparams.criterion}": loss},
        }

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": self.criterion(y_hat, y)}

    def validation_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {
            "val_loss": avg_loss,
            "log": {f"val/avg_{self.hparams.criterion}": avg_loss},
        }

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            TrendsNpSet(
                mode="train",
                config=self.hparams
            ),
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            TrendsNpSet(
                mode="val",
                config=self.hparams,
            ),
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    # fabric

    def get_net(self):
        if "conv3d_regressor" == self.hparams.net:
            return Conv3DRegressor()
        else:
            raise NotImplementedError("Not a valid model configuration.")

    def get_criterion(self):
        if "w_nae" == self.hparams.criterion:
            return WNAELoss()
        elif "l1" == self.hparams.criterion:
            return nn.L1Loss()
        else:
            raise NotImplementedError("Not a valid criterion configuration.")

    def get_optimizer(self) -> object:
        if "adam" == self.hparams.optimizer:
            return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        elif "sgd" == self.hparams.optimizer:
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.sgd_wd,
            )
        else:
            raise NotImplementedError("Not a valid optimizer configuration.")

    def get_scheduler(self, optimizer) -> object:
        if "plateau" == self.hparams.scheduler:
            return ReduceLROnPlateau(optimizer)
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")
