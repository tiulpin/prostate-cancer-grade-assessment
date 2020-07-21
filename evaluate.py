# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

import datetime
from argparse import ArgumentParser, Namespace

import torch
import torch.utils
import numpy as np

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm.auto import tqdm

from src.datasets.panda import PANDADataset
from src.transforms.tta import d4_tta
from src.pl_module import CoolSystem

SEED = 111
seed_everything(111)


def get_test_dataloder(hparams: Namespace) -> DataLoader:
    test_dataset = PANDADataset(mode=hparams.mode, config=hparams,)

    return DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )


def load_model(model_name: str, weights: str):
    model = CoolSystem.net_mapping(model_name, 5)
    model.load_state_dict(torch.load(
        weights, map_location=lambda storage, loc: storage), strict=True)
    model.eval()
    model.cuda()
    print('Loaded model {} from checkpoint {}'.format(model_name, weights))
    return model


def get_ground_truth(loader: DataLoader):
    gt = list()

    for _, y in loader:
        target = y.sum(1)
        gt.append(target)

    gt = torch.cat(gt).cpu().numpy()
    return gt


def run_predictions(model: torch.nn.Module,
                    loader: DataLoader,
                    precision: int = 16,
                    use_tta: bool = False):
    preds, preds_threshold = list(), list()
    tta_transforms = d4_tta()

    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader)):
            x = x.cuda()
            if precision == 16:
                x = x.half()

            if use_tta:
                tta_pred, tta_pred_threshold = list(), list()

                for tta in tta_transforms:
                    y_hat = model(tta.batch_augment(x))
                    pred = y_hat.sigmoid().sum(1).detach().round()
                    pred_threshold = (y_hat.sigmoid().detach() >= 0.5).sum(1)
                    tta_pred.append(pred)
                    tta_pred_threshold.append(pred_threshold)

                pred = torch.round(
                    (torch.stack(tta_pred).sum(0).double() / len(tta_transforms)))
                pred_threshold = torch.round(
                    (torch.stack(tta_pred_threshold).sum(0).double() / len(tta_transforms)))

            else:
                y_hat = model(x)
                pred = y_hat.sigmoid().sum(1).detach().round()
                pred_threshold = (y_hat.sigmoid().detach() >= 0.5).sum(1)

            preds.append(pred)
            preds_threshold.append(pred_threshold)

    preds = torch.cat(preds).cpu().numpy()
    preds_threshold = torch.cat(preds_threshold).cpu().numpy()
    return preds, preds_threshold


def main(hparams: Namespace):
    assert len(hparams.nets) == len(hparams.weights_paths), \
        'Please provide equal number of weights paths and model names'

    loader = get_test_dataloder(hparams)
    models = list()

    for model_name, weights_path in zip(hparams.nets, hparams.weights_paths):
        model = load_model(model_name, weights_path)
        if hparams.precision == 16:
            model.half()
        models.append(model)

    predictions, predictions_threshold = list(), list()
    gt_class = get_ground_truth(loader)

    for model in models:
        pred, pred_thr = run_predictions(
            model, loader, hparams.precision, hparams.test_time_aug)
        predictions.append(pred)
        predictions_threshold.append(pred_thr)

    if len(models) > 1:
        predictions = np.array(predictions).sum(axis=0) / len(models)
        predictions_threshold = np.array(predictions_threshold).sum(axis=0) / len(models)
    else:
        predictions = predictions[0]
        predictions_threshold = predictions_threshold[0]

    qwk = cohen_kappa_score(
        predictions.astype(int), gt_class, weights='quadratic')
    qwk_thr = cohen_kappa_score(
        predictions_threshold.astype(int), gt_class, weights='quadratic')

    print('QWK with sum strategy: {:.4f}'.format(qwk))
    print(confusion_matrix(gt_class, predictions.astype(int)))
    print('QWK with threshold strategy: {:.4f}'.format(qwk_thr))
    print(confusion_matrix(gt_class, predictions_threshold.astype(int)))


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--root_path", default="../input/prostate-cancer-grade-assessment"
    )
    parser.add_argument("--image_folder", default="train_images")
    parser.add_argument("--use_cleaned_data", default=True, type=bool)

    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--test_time_aug", default=False, type=bool)
    parser.add_argument('--mode', choices=['val', 'holdout'],
                        default='val', type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--weights_paths", nargs='+', required=True)
    parser.add_argument("--nets", nargs='+', required=True)

    parser.add_argument("--use_preprocessed_tiles", default=True, type=bool)
    parser.add_argument('--normalize', choices=['imagenet', 'own', 'none'],
                        default='imagenet', type=str)
    parser.add_argument("--tile_size", default=256, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_tiles", default=36, type=int)
    parser.add_argument("--random_tiles_order", default=False, type=bool)
    parser.add_argument("--tile_mode", default=0, type=int)

    args = parser.parse_args()
    main(args)
