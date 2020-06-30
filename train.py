# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import datetime
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, loggers, seed_everything

from src.pl_module import CoolSystem

SEED = 111
seed_everything(111)


def main(hparams):
    now = datetime.datetime.now().strftime("%d.%H")
    experiment_name = f"{now}_{hparams.net}_{hparams.criterion}_fold_{hparams.fold}"

    model = CoolSystem(hparams=hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.logger = loggers.TensorBoardLogger(f"logs/", name=experiment_name,)

    trainer.fit(model)

    # to make submission without lightning
    torch.save(model.net.state_dict(), f"weights/{experiment_name}.pth")


if __name__ == "__main__":
    # TODO: move configuration to *.yaml with Hydra
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--train_path",
        default="../input/trends-assessment-prediction/np_train/",
        type=str,
    )
    parser.add_argument(
        "--val_path",
        default="../input/trends-assessment-prediction/np_train/",
        type=str,
    )
    parser.add_argument("--root_path", default="../input/trends-assessment-prediction")

    parser.add_argument("--profiler", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--auto_lr_find", default=False, type=bool)

    parser.add_argument("--val_check_interval", default=0.95, type=float)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)

    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=30, type=int)
    parser.add_argument("--early_stop_callback", default=False, type=bool)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.add_argument("--benchmark", default=False, type=bool)

    parser.add_argument("--net", default="conv3d_regressor", type=str)
    parser.add_argument("--criterion", default="w_nae", type=str)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--scheduler", default="plateau", type=str)

    parser.add_argument("--sgd_momentum", default=0.9, type=float)
    parser.add_argument("--sgd_wd", default=7e-4, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)

    parser.add_argument("--permutation", default=False, type=bool)

    args = parser.parse_args()
    main(args)
