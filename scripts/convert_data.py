# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

import h5py as hp
import numpy as np
import pandas as pd

DF_PATH = "../input/trends-assessment-prediction/train_scores.csv"
HP_ROOT = "../input/trends-assessment-prediction/fMRI_train/"
NP_ROOT = "../input/trends-assessment-prediction/np_train/"


def main():
    df = pd.read_csv(DF_PATH)
    filenames = df["Id"].values

    for index in range(len(filenames)):
        with hp.File(f"{HP_ROOT}/{filenames[index]}.mat") as file:
            np.save(f"{NP_ROOT}/{filenames[index]}.npy", file["SM_feature"][()])


if __name__ == "__main__":
    main()
