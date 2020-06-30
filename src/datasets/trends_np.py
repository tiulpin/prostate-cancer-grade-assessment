# coding: utf-8
__author__ = "tiulpin: https://kaggle.com/tiulpin"

from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrendsNpSet(Dataset):
    def __init__(self, mode: str, config, targets=None):
        super().__init__()
        if targets is None:
            targets = [
                "age",
                "domain1_var1",
                "domain1_var2",
                "domain2_var1",
                "domain2_var2",
            ]
        self.mode = mode
        if "train" == mode:
            self.root = config.train_path
        elif "val" == mode:
            self.root = config.val_path
        else:
            raise NotImplementedError("Not implemented dataset configuration")

        self.permutation = config.permutation

        df = pd.read_csv(f"{config.root_path}/{mode}_{config.fold}.csv")
        self.filenames = df["Id"].values
        self.labels = df[targets].values

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Union[Tuple[torch.tensor], torch.tensor]:
        t = torch.FloatTensor(np.load(f"{self.root}/{self.filenames[index]}.npy"))
        if self.permutation:
            t = t.permute(3, 0, 1, 2)

        if self.mode == "test":
            return t

        return t, self.labels[index]
