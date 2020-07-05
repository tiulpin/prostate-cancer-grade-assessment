# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

from typing import Tuple, List

import skimage.io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PANDADataset(Dataset):

    def __init__(self, mode: str, config, transform=None):
        self.mode = mode
        if "train" == mode:
            self.root = config.train_path
        elif "val" == mode:
            self.root = config.val_path
        else:
            raise NotImplementedError("Not implemented dataset configuration")

        self.tile_size = config.tile_size
        self.image_size = config.image_size
        self.num_tiles = config.num_tiles
        self.random_tiles_order = config.random_tiles_order
        self.tile_mode = config.tile_mode

        self.df = pd.read_csv(f"{config.root_path}/{mode}_{config.fold}.csv")
        self.image_folder = f"{config.root_path}/{config.image_folder}"
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        img_id = row.image_id

        tiff_file = f"{self.image_folder}/{img_id}.tiff"
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, _ = self.get_tiles(image)

        idxes = np.random.choice(
            list(range(self.num_tiles)), self.num_tiles, replace=False) \
            if self.random_tiles_order else list(range(self.num_tiles))

        num_row_tiles = int(np.sqrt(self.num_tiles))
        result_size = self.image_size * num_row_tiles
        images = np.zeros((result_size, result_size, 3))

        for h in range(num_row_tiles):
            for w in range(num_row_tiles):
                i = h * num_row_tiles + w

                this_img = tiles[idxes[i]]['img'] if len(tiles) > idxes[i] \
                    else np.full((self.image_size, self.image_size, 3), 255)
                this_img = 255 - this_img

                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']

                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1 + self.image_size, w1:w1 + self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']

        images = images.astype(np.float32) / 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return torch.tensor(images), torch.tensor(label)

    def get_tiles(self, image: np.ndarray) -> Tuple[List, bool]:
        result = []
        height, width, _ = image.shape

        pad_h = (self.tile_size - height % self.tile_size) % self.tile_size + \
                (self.tile_size * self.tile_mode // 2)
        pad_w = (self.tile_size - width % self.tile_size) % self.tile_size + \
                (self.tile_size * self.tile_mode // 2)

        padding = [[pad_h // 2, pad_h - pad_h // 2],
                   [pad_w // 2, pad_w - pad_w // 2], [0, 0]]
        image2 = np.pad(image, padding, constant_values=255)

        image3 = image2.reshape(
            image2.shape[0] // self.tile_size, self.tile_size,
            image2.shape[1] // self.tile_size, self.tile_size, 3)
        image3 = image3.transpose(0, 2, 1, 3, 4).reshape(
            -1, self.tile_size, self.tile_size, 3)

        num_tiles_with_info = (image3.reshape(image3.shape[0], -1).sum(1) <
                               self.tile_size ** 2 * 3 * 255).sum()

        if len(image3) < self.num_tiles:
            padding = [[0, self.num_tiles - len(image3)], [0, 0], [0, 0], [0, 0]]
            image3 = np.pad(image3, padding, constant_values=255)

        indexes = np.argsort(image3.reshape(image3.shape[0], -1).sum(-1))
        image3 = image3[indexes[:self.num_tiles]]

        for i in range(len(image3)):
            result.append({'img': image3[i], 'idx': i})

        return result, num_tiles_with_info >= self.num_tiles
