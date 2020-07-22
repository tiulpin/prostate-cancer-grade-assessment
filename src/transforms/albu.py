import albumentations as A


def get_individual_transforms():
    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.Transpose(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                    A.NoOp(),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(p=1.0),
                    A.NoOp(),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.ISONoise(p=1.0),
                    A.CoarseDropout(p=1.0, max_holes=16, max_height=16, max_width=16),
                    A.NoOp(),
                ],
                p=1.0,
            ),
        ]
    )

    return transforms


def get_global_transforms():
    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.Transpose(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                    A.NoOp(),
                ],
                p=1.0,
            ),
        ]
    )

    return transforms


def get_simple_train_transforms():
    transforms = A.Compose(
        [A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),]
    )

    return transforms


def get_val_transforms():
    transforms = A.Compose([])

    return transforms
