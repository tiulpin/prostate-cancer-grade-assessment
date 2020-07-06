# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import datetime
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, loggers, seed_everything

from src.pl_module import CoolSystem

SEED = 111
seed_everything(111)


def main(hparams: Namespace):
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
        "--root_path", default="../input/prostate-cancer-grade-assessment"
    )
    parser.add_argument("--image_folder", default="train_images")

    parser.add_argument("--profiler", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--auto_lr_find", default=False, type=bool)

    parser.add_argument("--val_check_interval", default=0.95, type=float)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)

    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--gpus", default=6, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=30, type=int)
    parser.add_argument("--early_stop_callback", default=False, type=bool)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.add_argument("--benchmark", default=False, type=bool)

    parser.add_argument("--net", default="effnet_b0", type=str)
    parser.add_argument("--output_dim", default=5, type=int)
    parser.add_argument("--criterion", default="bce_with_logits", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--scheduler", default="cyclic", type=str)

    parser.add_argument("--sgd_momentum", default=0.9, type=float)
    parser.add_argument("--sgd_wd", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=0.01, type=float)

    parser.add_argument("--imagenet_norm", default=True, type=bool)
    parser.add_argument("--tile_size", default=256, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_tiles", default=36, type=int)
    parser.add_argument("--random_tiles_order", default=True, type=bool)
    parser.add_argument("--tile_mode", default=0, type=int)

    args = parser.parse_args()
    main(args)