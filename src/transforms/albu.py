import albumentations


def get_train_transforms():
    transforms = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
    ])

    return transforms


def get_val_transforms():
    transforms = albumentations.Compose([])

    return transforms
