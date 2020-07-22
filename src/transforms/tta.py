from itertools import product

import torch


class ProstateTTA:
    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError


class HorizontalFlipTTA(ProstateTTA):
    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)


class VerticalFlipTTA(ProstateTTA):
    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)


class Rotate90TTA(ProstateTTA):
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))


class ComposeTTA(ProstateTTA):
    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images


def d4_tta():
    tta_transforms = [
        ComposeTTA([]),
        ComposeTTA([Rotate90TTA()]),
        ComposeTTA([Rotate90TTA(), Rotate90TTA()]),
        ComposeTTA([Rotate90TTA(), Rotate90TTA(), Rotate90TTA()]),
        ComposeTTA([HorizontalFlipTTA()]),
        ComposeTTA([HorizontalFlipTTA(), Rotate90TTA()]),
        ComposeTTA([HorizontalFlipTTA(), Rotate90TTA(), Rotate90TTA()]),
        ComposeTTA([HorizontalFlipTTA(), Rotate90TTA(), Rotate90TTA(), Rotate90TTA()]),
    ]

    return tta_transforms
