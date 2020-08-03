# coding: utf-8
from src.datasets.panda import get_tiles

__author__ = "sevakon: https://kaggle.com/sevakon"

import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

sys.path.append(".")


def welford_algo(image: np.ndarray, mean: list, m2: list, num_pixels: int):
    """
    Calculating mean and standard deviation
    for red, green, blue channels in image dataset
    with Welford's online algorithm
    """

    red = image[:, :, 0].flatten().tolist()
    green = image[:, :, 1].flatten().tolist()
    blue = image[:, :, 2].flatten().tolist()

    for (r, g, b) in zip(red, green, blue):
        num_pixels += 1

        delta_red = r - mean[0]
        delta_green = g - mean[1]
        delta_blue = b - mean[2]

        mean[0] += delta_red / num_pixels
        mean[1] += delta_green / num_pixels
        mean[2] += delta_blue / num_pixels

        m2[0] += delta_red * (r - mean[0])
        m2[1] += delta_green * (g - mean[1])
        m2[2] += delta_green * (b - mean[2])

    return mean, m2, num_pixels


def main(config: Namespace):
    train_data = pd.read_csv(f"{config.root_path}/train_{config.fold}.csv")

    mean = np.array([0.0, 0.0, 0.0])
    m2 = np.array([0.0, 0.0, 0.0])
    num_pixels = 0

    for img_id in tqdm(train_data.image_id):
        if config.use_preprocessed:
            npy_file = f"{config.root_path}/" f"{config.image_folder}/{img_id}.npy"
            tiles = np.load(npy_file)

        else:
            tiff_file = f"{config.root_path}/" f"{config.image_folder}/{img_id}.tiff"
            image = skimage.io.MultiImage(tiff_file)[1]
            tiles, _ = get_tiles(image, config.tile_size, config.num_tiles)

        idxes = list(range(config.num_tiles))

        num_row_tiles = int(np.sqrt(config.num_tiles))
        result_size = config.image_size * num_row_tiles
        images = np.zeros((result_size, result_size, 3))

        for h in range(num_row_tiles):
            for w in range(num_row_tiles):
                i = h * num_row_tiles + w

                this_img = (
                    tiles[idxes[i]] if len(tiles) > idxes[i] else np.full(
                        (config.image_size, config.image_size, 3), 255))
                this_img = 255 - this_img

                h1 = h * config.image_size
                w1 = w * config.image_size
                images[h1:h1 + config.image_size,
                       w1:w1 + config.image_size] = this_img

        images = images.astype(np.float32) / 255

        mean, m2, num_pixels = welford_algo(images, mean, m2, num_pixels)

    std = np.sqrt(m2 / (num_pixels - 1))

    stats_df = pd.DataFrame.from_dict({"mean": mean, "std": std})
    stats_df.to_csv(f"{config.root_path}/stats_{config.fold}.csv")
    print(stats_df)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--root_path",
                        default="../input/prostate-cancer-grade-assessment")
    parser.add_argument("--image_folder", default="train_images")
    parser.add_argument("--use_preprocessed", default=True, type=bool)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_tiles", default=36, type=int)

    args = parser.parse_args()
    main(args)
