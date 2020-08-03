# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_PATH = "../input/prostate-cancer-grade-assessment"
SEED = 42
FOLDS = 5


def main():
    data = pd.read_csv(f"{ROOT_PATH}/train_cleaned.csv")

    train_data, holdout_data = train_test_split(data,
                                                test_size=0.15,
                                                random_state=SEED,
                                                shuffle=True,
                                                stratify=data.isup_grade)

    train_data = train_data.reset_index(drop=True)
    holdout_data = holdout_data.reset_index(drop=True)

    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(
            skf.split(train_data, train_data["isup_grade"])):
        train_data.loc[trn_idx].to_csv(f"{ROOT_PATH}/train_cleaned_{fold}.csv",
                                       index=False)
        train_data.loc[val_idx].to_csv(f"{ROOT_PATH}/val_cleaned_{fold}.csv",
                                       index=False)

    holdout_data.to_csv(f"{ROOT_PATH}/holdout.csv", index=False)


if __name__ == "__main__":
    main()
