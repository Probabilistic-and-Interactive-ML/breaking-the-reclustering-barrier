import numpy as np
import torch
from torchvision.transforms import v2

from config.base_config import Args
from src.deep._data_utils import get_dataloader


def get_normalize_fn(data: torch.Tensor):
    # calculation of means and stds assumes images with color channel
    channel_means = data.mean([0, 2, 3])
    channel_stds = data.std([0, 2, 3])
    # preprocessing functions
    normalize_fn = v2.Normalize(channel_means, channel_stds)
    return normalize_fn


def preprocess_data(data: np.array, resize_shape: int, flatten: bool, args: Args):
    if len(data.shape) == 3:
        # greyscale data has no color channel
        data = np.expand_dims(data, axis=1)
    transform_list = [
        # original data needs to be of the same size as augmented data
        v2.Resize(size=(resize_shape, resize_shape), antialias=None),
        get_normalize_fn(torch.from_numpy(data)),
    ]
    if flatten:
        flatten_fn = v2.Lambda(torch.flatten)
        transform_list.append(flatten_fn)

    orig_transforms = v2.Compose(transform_list)
    dl = get_dataloader(data, batch_size=args.batch_size, shuffle=False, ds_kwargs={"orig_transforms_list": [orig_transforms]})
    preprocessed_data = torch.cat([_b[1] for _b in dl]).numpy()
    return preprocessed_data


def get_dataloader_with_augmentation(
    data: np.ndarray,
    batch_size: int,
    flatten: bool,
    data_name: str,
    crop_size: int,
    augment_both_views: bool = False,
    resize_shape: int = 28,
    num_workers: int = 1,
):
    data = torch.tensor(data)
    data /= 255.0

    if len(data.shape) == 3:
        # add color channel for greyscale images without color channel
        data = data.unsqueeze(1)

    normalize_fn = get_normalize_fn(data)
    # augmentation transforms for color images
    if data_name in {"cifar10", "gtsrb", "cifar100_20"}:
        print("Use Color Augmentation")
        # Setting taken from https://github.com/idstcv/SeCu/blob/7d5ae754d5d0c1374e9b7f32612042b06dfc749c/main.py#L261
        min_crop = 0.08
        transform_list = [
            v2.RandomResizedCrop(crop_size, scale=(min_crop, 1.0), antialias=None),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            # NOT used by SimCLR for CIFAR10
            # v2.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            v2.RandomHorizontalFlip(),
            # threshold taken from BYOL paper
            v2.RandomSolarize(threshold=0.5, p=0.2),
            normalize_fn,
        ]
        orig_transform_list = [
            # original data needs to be of the same size as augmented data
            v2.Resize(size=(crop_size, crop_size), antialias=None),
            normalize_fn,
        ]

    # augmentation transforms for grayscale images
    else:
        print("Use Grayscale Augmentation")
        transform_list = [
            # use the same resize shape for all grayscale images
            v2.Resize(size=(resize_shape, resize_shape), antialias=None),
            v2.RandomAffine(degrees=(-16, +16), translate=(0.1, 0.1), shear=(-8, 8), fill=0),
            normalize_fn,
        ]
        orig_transform_list = [
            # original data needs to be of the same size as augmented data
            v2.Resize(size=(resize_shape, resize_shape), antialias=None),
            normalize_fn,
        ]

    if flatten:
        flatten_fn = v2.Lambda(torch.flatten)
        transform_list.append(flatten_fn)
        orig_transform_list.append(flatten_fn)
    aug_transforms = v2.Compose(transform_list)
    orig_transforms = v2.Compose(orig_transform_list)

    if augment_both_views:
        ds_kwargs = {"aug_transforms_list": [aug_transforms], "orig_transforms_list": [aug_transforms]}
    else:
        ds_kwargs = {"aug_transforms_list": [aug_transforms], "orig_transforms_list": [orig_transforms]}

    # convert back to numpy
    data = data.numpy()
    # pass transforms to dataloader
    aug_dl = get_dataloader(
        data,
        batch_size=batch_size,
        shuffle=True,
        ds_kwargs=ds_kwargs,
        drop_last=True,
        dl_kwargs={
            "num_workers": num_workers,
            "persistent_workers": True if num_workers > 1 else False,
            "pin_memory": True,
        },
    )
    orig_dl = get_dataloader(data, batch_size=batch_size, shuffle=False, ds_kwargs={"orig_transforms_list": [orig_transforms]})
    return aug_dl, orig_dl
