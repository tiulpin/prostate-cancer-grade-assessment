# coding: utf-8
__author__ = "sevakon: https://kaggle.com/sevakon"

from argparse import ArgumentParser, Namespace
from tqdm import tqdm

import pandas as pd
import numpy as np
import skimage.io
import sys

sys.path.append('.')

from src.datasets.panda import get_tiles


def main(config: Namespace):
    train_data = pd.read_csv(f"{config.root_path}/train.csv")
    image_ids = train_data.image_id

    for img_id in tqdm(image_ids):
        tiff_file = f"{config.root_path}/{config.image_folder}/{img_id}.tiff"
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, _ = get_tiles(image, config.tile_size, config.num_tiles)
        npy_file = f"{config.root_path}/{config.image_folder}/{img_id}.npy"
        np.save(npy_file, tiles)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--root_path", default="../input/prostate-cancer-grade-assessment"
    )
    parser.add_argument("--image_folder", default="train_images")
    parser.add_argument("--tile_size", default=256, type=int)
    parser.add_argument("--num_tiles", default=36, type=int)
    parser.add_argument("--tile_mode", default=0, type=int)

    args = parser.parse_args()

    main(args)