# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT_PATH = "../input/trends-assessment-prediction/"
SEED = 111
FOLDS = 5


def main():
    train_data = pd.read_csv(f"{ROOT_PATH}/train_scores.csv").dropna()
    train_data.index = np.arange(train_data.shape[0])
    train_data = train_data.iloc[
        train_data.drop("Id", axis=1).drop_duplicates().index.values
    ]
    train_data.index = np.arange(train_data.shape[0])
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_data)):
        print(fold)
        train_data.loc[trn_idx].to_csv(f"{ROOT_PATH}/train_{fold}.csv", index=False)
        train_data.loc[val_idx].to_csv(f"{ROOT_PATH}/val_{fold}.csv", index=False)


if __name__ == "__main__":
    main()
