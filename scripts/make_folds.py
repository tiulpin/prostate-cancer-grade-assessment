# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT_PATH = "../input/prostate-cancer-grade-assessment"
SEED = 42
FOLDS = 5


def main():
    train_data = pd.read_csv(f"{ROOT_PATH}/train.csv")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(
            skf.split(train_data, train_data["isup_grade"])):
        train_data.loc[trn_idx].to_csv(f"{ROOT_PATH}/train_{fold}.csv",
                                       index=False)
        train_data.loc[val_idx].to_csv(f"{ROOT_PATH}/val_{fold}.csv",
                                       index=False)


if __name__ == "__main__":
    main()
