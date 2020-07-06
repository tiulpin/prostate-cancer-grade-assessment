# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"


import pytorch_lightning as pl
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from src.datasets.panda import PANDADataset
from src.models.networks.effnet_regressor import EffNetRegressor
from src.transforms.albu import get_train_transforms, get_val_transforms


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self.val_df = None

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        train_step = {
            "loss": loss,
            "log": {f"train/{self.hparams.criterion}": loss},
        }

        return train_step

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        preds = y_hat.sigmoid().sum(1).detach().round()
        targets = y.sum(1)

        val_step = {
            "val_loss": self.criterion(y_hat, y),
            "preds": preds,
            "targets": targets
        }

        return val_step

    def validation_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        predictions = torch.stack([x["preds"] for x in outputs]).cpu().numpy()
        targets = torch.stack([x["targets"] for x in outputs]).cpu().numpy()

        acc = (predictions == targets).mean() * 100.
        qwk = cohen_kappa_score(predictions, targets, weights='quadratic')
        qwk_karolinska, qwk_radbound = qwk, qwk

        # calculate metrics for different data centers
        if self.val_df is not None:
            karolinska = self.val_df['data_provider'] == 'karolinska'
            qwk_karolinska = cohen_kappa_score(
                predictions[karolinska],
                self.val_df[karolinska].isup_grade.values,
                weights='quadratic')

            radbound = self.val_df['data_provider'] == 'radbound'
            qwk_radbound = cohen_kappa_score(
                predictions[radbound],
                self.val_df[radbound].isup_grade.values,
                weights='quadratic')

        val_epoch_end = {
            "val_loss": avg_loss,
            "acc": acc,
            "qwk": qwk,
            "qwk_karolinska": qwk_karolinska,
            "qwk_radbound": qwk_radbound,
            "log": {
                f"val/avg_{self.hparams.criterion}": avg_loss,
                "val/acc": acc,
                "val/qwk": qwk,
                "val/qwk_karolinska": qwk_karolinska,
                "val/qwk_radbound": qwk_radbound,
            }
        }

        return val_epoch_end

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = PANDADataset(
                mode="train",
                config=self.hparams,
                transform=get_train_transforms()
            )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = PANDADataset(
                mode="val",
                config=self.hparams,
                transform=get_val_transforms()
            )

        self.val_df = val_dataset.df

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def get_net(self):
        if self.hparams.net == "effnet_b0":
            return EffNetRegressor(
                'efficientnet_b0', self.hparams.output_dim)

        elif self.hparams.net == "effnet_b1":
            return EffNetRegressor(
                'efficientnet_b1', self.hparams.output_dim)

        elif self.hparams.net == "effnet_b2":
            return EffNetRegressor(
                'efficientnet_b2', self.hparams.output_dim)

        elif self.hparams.net == "effnet_b3":
            return EffNetRegressor(
                'efficientnet_b3', self.hparams.output_dim)

        elif self.hparams.net == "tf_effnet_b4":
            return EffNetRegressor(
                'tf_efficientnet_b4_ns', self.hparams.output_dim)

        elif self.hparams.net == "tf_effnet_b5":
            return EffNetRegressor(
                'tf_efficientnet_b5_ns', self.hparams.output_dim)

        elif self.hparams.net == "tf_effnet_b6":
            return EffNetRegressor(
                'tf_efficientnet_b6_ns', self.hparams.output_dim)

        elif self.hparams.net == "tf_effnet_b7":
            return EffNetRegressor(
                'tf_efficientnet_b7_ns', self.hparams.output_dim)

        else:
            raise NotImplementedError("Not a valid model configuration.")

    def get_criterion(self):
        if "l1" == self.hparams.criterion:
            return nn.L1Loss()
        elif "bce_with_logits" == self.hparams.criterion:
            return nn.BCEWithLogitsLoss()
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
        elif "cyclic" == self.hparams.scheduler:
            return CyclicLR(optimizer,
                            base_lr=self.learning_rate / 100,
                            max_lr=self.learning_rate,
                            step_size_up=4000 / self.batch_size)
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")
