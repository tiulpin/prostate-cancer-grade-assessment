# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon & tiulpin: https://kaggle.com/sevakon"

import pytorch_lightning as pl
import torch
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from src.datasets.panda import PANDADataset
from src.models.networks.effnet import EffNetDoubleClassifier, EffNetClassifier
from src.models.networks.resnext import ResNeXtClassifier
from src.schedulers.warmup import GradualWarmupScheduler
from src.transforms.albu import get_global_transforms, get_individual_transforms


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.BATCH_SIZE

        self.val_df = None

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        train_step = {
            "loss": loss,
            "log": {
                f"train/{self.hparams.criterion}": loss
            },
        }

        return train_step

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, y = batch
        y_hat = self.forward(x)
        preds = y_hat.sigmoid().sum(1).detach().round()
        preds_threshold = (y_hat.sigmoid().detach() >= 0.5).sum(1)
        targets = y.sum(1)

        val_step = {
            "val_loss": self.criterion(y_hat, y),
            "preds": preds,
            "preds_threshold": preds_threshold,
            "targets": targets,
        }

        return val_step

    def validation_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        predictions = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        predictions_threshold = (torch.cat(
            [x["preds_threshold"] for x in outputs]).cpu().numpy())
        targets = torch.cat([x["targets"] for x in outputs]).cpu().numpy()

        acc = (predictions == targets).mean() * 100.0
        acc_threshold = (predictions_threshold == targets).mean() * 100.0

        qwk = cohen_kappa_score(predictions, targets, weights="quadratic")
        qwk_threshold = cohen_kappa_score(predictions_threshold,
                                          targets,
                                          weights="quadratic")

        qwk_karolinska, qwk_radbound = 0, 0

        # calculate metrics for different data centers
        if self.val_df is not None:
            karolinska = self.val_df["data_provider"] == "karolinska"
            if len(karolinska) == len(outputs):
                qwk_karolinska = cohen_kappa_score(
                    predictions[karolinska],
                    self.val_df[karolinska].isup_grade.values,
                    weights="quadratic",
                )

            radbound = self.val_df["data_provider"] == "radbound"
            if len(radbound) == len(outputs):
                qwk_radbound = cohen_kappa_score(
                    predictions[radbound],
                    self.val_df[radbound].isup_grade.values,
                    weights="quadratic",
                )

        monitor_qwk = torch.tensor(max(qwk, qwk_threshold))
        monitor_acc = torch.tensor(max(acc, acc_threshold))

        val_epoch_end = {
            "val_loss": avg_loss,
            "acc": monitor_acc,
            "qwk": monitor_qwk,
            "log": {
                f"val/avg_{self.hparams.criterion}": avg_loss,
                "val/acc": acc,
                "val/acc_threshold": acc_threshold,
                "val/qwk": qwk,
                "val/qwk_threshold": qwk_threshold,
                "val/qwk_karolinska": qwk_karolinska,
                "val/qwk_radbound": qwk_radbound,
            },
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
            individual_transform=get_individual_transforms(),
            global_transform=get_global_transforms(),
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = PANDADataset(
            mode="val",
            config=self.hparams,
        )

        self.val_df = val_dataset.df

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def net_mapping(model_name: str, output_dim: int = 5) -> torch.nn.Module:
        if model_name == "effnet_b0":
            return EffNetClassifier("efficientnet_b0", output_dim)
        elif model_name == "double_effnet_b0":
            return EffNetDoubleClassifier("efficientnet_b0", output_dim)
        elif model_name == "effnet_b1":
            return EffNetClassifier("efficientnet_b1", output_dim)
        elif model_name == "double_effnet_b1":
            return EffNetDoubleClassifier("efficientnet_b1", output_dim)
        elif model_name == "effnet_b2":
            return EffNetClassifier("efficientnet_b2", output_dim)
        elif model_name == "double_effnet_b2":
            return EffNetDoubleClassifier("efficientnet_b2", output_dim)
        elif model_name == "effnet_b3":
            return EffNetClassifier("efficientnet_b3", output_dim)
        elif model_name == "double_effnet_b3":
            return EffNetDoubleClassifier("efficientnet_b3", output_dim)
        elif model_name == "effnet_b4":
            return EffNetClassifier("tf_efficientnet_b4_ns", output_dim)
        elif model_name == "double_effnet_b4":
            return EffNetDoubleClassifier("tf_efficientnet_b4_ns", output_dim)
        elif model_name == "effnet_b5":
            return EffNetClassifier("tf_efficientnet_b5_ns", output_dim)
        elif model_name == "double_effnet_b5":
            return EffNetDoubleClassifier("tf_efficientnet_b5_ns", output_dim)
        elif model_name == "effnet_b6":
            return EffNetClassifier("tf_efficientnet_b6_ns", output_dim)
        elif model_name == "double_effnet_b6":
            return EffNetDoubleClassifier("tf_efficientnet_b6_ns", output_dim)
        elif model_name == "effnet_b7":
            return EffNetClassifier("tf_efficientnet_b7_ns", output_dim)
        elif model_name == "double_effnet_b7":
            return EffNetDoubleClassifier("tf_efficientnet_b7_ns", output_dim)
        elif model_name == "resnext50":
            return ResNeXtClassifier("resnext50_32x4d", output_dim)
        else:
            raise NotImplementedError("Not a valid model configuration.")

    def get_net(self) -> torch.nn.Module:
        return CoolSystem.net_mapping(self.hparams.net,
                                      self.hparams.output_dim)

    def get_criterion(self):
        if "l1" == self.hparams.criterion:
            return torch.nn.L1Loss()
        elif "bce_with_logits" == self.hparams.criterion:
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Not a valid criterion configuration.")

    def get_optimizer(self) -> object:
        if "adam" == self.hparams.optimizer:
            return torch.optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(self.net.parameters(),
                                     lr=self.learning_rate)
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
        elif "plateau+warmup" == self.hparams.scheduler:
            plateau = ReduceLROnPlateau(optimizer)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=plateau,
            )
        elif "cyclic" == self.hparams.scheduler:
            return CyclicLR(
                optimizer,
                base_lr=self.learning_rate / 100,
                max_lr=self.learning_rate,
                step_size_up=4000 / self.batch_size,
            )
        elif "cosine" == self.hparams.scheduler:
            return CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        elif "cosine+warmup" == self.hparams.scheduler:
            cosine = CosineAnnealingLR(
                optimizer,
                self.hparams.max_epochs - self.hparams.warmup_epochs)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=cosine,
            )
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")
