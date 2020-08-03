# coding: utf-8
from typing import List
from collections import OrderedDict
from argparse import ArgumentParser

__author__ = "sevakon: https://kaggle.com/sevakon"

import sys

import torch

sys.path.append(".")


def average_weights(state_dicts: List[dict]):
    average_dict = OrderedDict()

    # check that the keys inside dictionaries are equal
    last_keys = None
    for dictionary in state_dicts:
        if last_keys is None:
            last_keys = dictionary.keys()
            continue

        if last_keys != dictionary.keys():
            raise AssertionError(
                "Provided checkpoints have different set of keys, check your models"
            )

        last_keys = dictionary.keys()

    for k in state_dicts[0].keys():
        average_dict[k] = sum([state_dict[k] for state_dict in state_dicts
                               ]) / float(len(state_dicts))
    return average_dict


def main(checkpoint_paths: List[str], save_to_path: str):
    if len(checkpoint_paths) <= 1:
        raise AssertionError(
            "Please provide more than 1 checkpoints to average")

    state_dicts = [torch.load(path) for path in checkpoint_paths]
    dict_to_save = average_weights(state_dicts)

    torch.save(dict_to_save, save_to_path)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--checkpoint_paths", nargs="+", required=True)
    parser.add_argument("--save_to_path", type=str, required=True)

    args = parser.parse_args()
    main(args.checkpoint_paths, args.save_to_path)
