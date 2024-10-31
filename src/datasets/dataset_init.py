import numpy as np
import torch
import torchvision

from config.base_config import Args
from src.datasets.augmentation_dataloader import get_dataloader_with_augmentation, preprocess_data
from src.datasets.load_datasets import (
    load_cifar10,
    load_cifar100_20,
    load_fmnist,
    load_gtsrb,
    load_kmnist,
    load_mnist,
    load_optdigits,
    load_usps,
)
from src.deep._data_utils import get_dataloader


def _get_smallest_subset(dset_name: str, subset: str) -> str:
    """Gets the smallest subset for a dataset"""

    if subset != "smallest":
        return subset
    elif dset_name == "cifar10" and subset == "smallest":
        return "test"
    elif dset_name == "cifar100_20" and subset == "smallest":
        return "test"
    elif dset_name == "mnist" and subset == "smallest":
        return "test"
    elif dset_name == "optdigits" and subset == "smallest":
        return "all"
    elif dset_name == "usps" and subset == "smallest":
        return "all"
    elif dset_name == "fmnist" and subset == "smallest":
        return "test"
    elif dset_name == "kmnist" and subset == "smallest":
        return "test"
    elif dset_name == "gtsrb" and subset == "smallest":
        return "all"
    else:
        raise ValueError(f"Invalid dataset {dset_name} specified.")


def load_dataset(
    args: Args,
    downloads_path: str,
    normalize_channels: bool = True,
    flatten: bool = True,
    subset: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the chosen data set."""

    # Subset from arguments can be overwritten, e.g., to get the test subset for evaluation
    if subset is None:
        subset = args.dataset_subset

    if args.dataset_name == "cifar10":
        data, labels = load_cifar10(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "cifar100_20":
        data, labels = load_cifar100_20(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "mnist":
        data, labels = load_mnist(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "optdigits":
        # normalize_channels is currently not implemented for optdigits and throws an error
        data, labels = load_optdigits(
            flatten=flatten,
            downloads_path=downloads_path,
            subset=subset,
        )
    elif args.dataset_name == "usps":
        data, labels = load_usps(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "fmnist":
        data, labels = load_fmnist(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "kmnist":
        data, labels = load_kmnist(
            flatten=flatten,
            downloads_path=downloads_path,
            normalize_channels=normalize_channels,
            subset=subset,
        )
    elif args.dataset_name == "gtsrb":
        data, labels = load_gtsrb(
            flatten=False,
            downloads_path=downloads_path,
            normalize_channels=False,
            subset=subset,
        )
        # Use subset of GTSRB as in DipDECK paper of 5 different signs resulting in 7860 images.
        # {2: 'Speed limit (50km/h), 12: 'Priority road', 17: 'No entry', 18: 'General caution', 35: 'Ahead only'}
        # TODO: Remove
        # alternatively we could use the selection of the ACe/DeC paper with 10 classes: 2,8,10,11,12,13,14,17,18,35
        data, labels = get_label_subset(data, labels, label_subset=[2, 12, 17, 18, 35], relabel=True)

        # Clustpy has not implemented histogram equalization, thus we need to do it by hand:
        # Apply histogram equalization due to low contrast in GTSRB images
        data = torchvision.transforms.functional.equalize(torch.from_numpy(data).to(torch.uint8))
        # convert data to torch.tensor float
        data = data.float()
        # scale data from 0-255 to 0 - 1
        data /= 255.0
        if normalize_channels:
            # calculate mean and standard deviation for the 3 RGB channels
            mean = data.mean([0, 2, 3])
            std = data.std([0, 2, 3])
            # normalize channels
            data -= mean.reshape(1, 3, 1, 1)
            data /= std.reshape(1, 3, 1, 1)
        if flatten:
            # flatten data
            data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
        # convert to to numpy
        data = data.numpy()
    else:
        raise ValueError(f"Invalid dataset {args.dataset_name} specified.")

    if args.convnet != "none":
        # convnets require non-flattened data
        assert len(data.shape) > 2

        if len(data.shape) == 3:
            # greyscale data has no color channel
            data = np.expand_dims(data, axis=1)

        # convnets require non-flattened data (batch_dim x channels x height x width) with 4 axes and 3 color channels
        assert len(data.shape) == 4

        # multiply color channel for greyscale data
        if data.shape[1] == 1:
            data = np.repeat(data, 3, axis=1)
        # make sure we have 3 color channels
        assert data.shape[1] == 3

    return data, labels


def get_label_subset(data: np.array, labels: np.array, label_subset: list, relabel: bool = False) -> tuple[np.array, np.array]:
    """Returns subset of selected labels.

    Parameters
    ----------
    data : np.array
        data set
    labels : np.array
        ground truth labels
    label_subset: list[int]
        list of class labels that should be selected, e.g.,
        [0, 1, 5], will select all instances that have labels 0, 1 and 5.
    relabel: bool
        If True selected labels will be relabeled to start from 0 to len(label_subset).
        The relabeling is done in ascending manner, e.g, for label_subset=[1,0,5] will
        be mapped to {0:0, 1:1, 5:2}. (default: False)

    Returns
    -------
    data_subset, labels_subset : (np.ndarray, np.ndarray)
        the subsets of the data and labels numpy array

    Raises
    -------
        ValueError
            If a label in the provided label_subset is not part of the passed labels
    """
    labels_new = []
    data_new = []
    all_unique_labels = set(labels.tolist())
    for label_i in label_subset:
        if label_i not in all_unique_labels:
            raise ValueError(f"label {label_i} is not contained in provided labels.")
        mask_i = labels == label_i
        labels_new.append(labels[mask_i].copy())
        data_new.append(data[mask_i].copy())
    data_new = np.concatenate(data_new)
    labels_new = np.concatenate(labels_new)

    if relabel:
        # get unique labels
        unique_labels = list(set(labels_new.tolist()))
        # sort labels ascendingly
        unique_labels.sort()
        # relabel the labels from 0 to len(unique_labels)
        relabel_dict = {idx: i for i, idx in enumerate(unique_labels)}
        labels_new = np.array([relabel_dict[i] for i in labels_new])
    return data_new, labels_new


def _get_train_eval_loader_with_augmentation(args: Args, data_path: str, flatten_data: bool, resize_shape: int) -> tuple:
    # For augmentation to properly work we need the unnormalized and non-flattened data
    data_unnormalized, labels = load_dataset(
        args, downloads_path=f"{data_path}/{args.dataset_name}", normalize_channels=False, flatten=False
    )
    # Returns a tuple with (dataloader_with_augmentation, dataloader_without_augmentation).

    train_dl, orig_dl = get_dataloader_with_augmentation(
        data=data_unnormalized,
        batch_size=args.batch_size,
        flatten=flatten_data,
        data_name=args.dataset_name,
        crop_size=args.crop_size,
        augment_both_views=args.augment_both_views,
        num_workers=args.experiment.num_workers_dataloader,
        resize_shape=resize_shape,
    )
    # Preprocess data with transforms specified in orig_dl as data to save compute during training
    data = torch.cat([_b[1] for _b in orig_dl]).numpy()
    eval_dl = get_dataloader(
        data, batch_size=args.batch_size, shuffle=False, dl_kwargs={"num_workers": args.experiment.num_workers_dataloader}
    )

    return train_dl, eval_dl, data, labels


def _get_train_eval_loader(
    args: Args, data_path: str, flatten_data: bool, resize_shape: int, dataset_subset: str = None
) -> tuple:
    if dataset_subset is None:
        dataset_subset = args.dataset_subset
    if args.convnet != "none" or args.augmented_pretraining or args.augmentation_invariance:
        # if convnet is used or any augmentation is used, than we need to resize the data
        data_unnormalized, labels = load_dataset(
            args,
            downloads_path=f"{data_path}/{args.dataset_name}",
            normalize_channels=False,
            flatten=False,
            subset=dataset_subset,
        )
        # Resize, normalize and optionally flatten
        data = preprocess_data(data_unnormalized, resize_shape, flatten_data, args)
    else:
        data, labels = load_dataset(
            args=args, downloads_path=f"{data_path}/{args.dataset_name}", flatten=flatten_data, subset=dataset_subset
        )

    train_dl = get_dataloader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        dl_kwargs={
            "num_workers": args.experiment.num_workers_dataloader,
            "persistent_workers": True if args.experiment.num_workers_dataloader > 1 else False,
            "pin_memory": True,
        },
    )
    eval_dl = get_dataloader(
        data, batch_size=args.batch_size, shuffle=False, dl_kwargs={"num_workers": args.experiment.num_workers_dataloader}
    )

    return train_dl, eval_dl, data, labels


def get_train_eval_test_dataloaders(args, data_path):
    """Returns ae_train_dl, ae_eval_dl, dc_train_dl, dc_eval_dl, test_dl, data, labels, test_data, test_labels"""
    # handling data preprocessing depending on the used architecture
    if args.convnet == "none":
        flatten_data = True
        if args.dataset_name in ["cifar10", "gtsrb", "cifar100_20"]:
            resize_shape = 32
        else:
            # resize_shape for greyscale transforms (MNIST shapes are default)
            resize_shape = 28
    else:
        flatten_data = False
        # resize_shape for greyscale transforms (32x32 is default for convnets)
        resize_shape = args.crop_size

    # Load and preprocess data
    if args.augmentation_invariance or args.augmented_pretraining:
        # If any augmentation is used we need the corresponding dataloaders
        aug_train_dl, aug_eval_dl, data, labels = _get_train_eval_loader_with_augmentation(
            args, data_path, flatten_data, resize_shape
        )
    if not args.augmentation_invariance or not args.augmented_pretraining:
        # If any stage (pretraining or deep clustering) does not use augmentation we need the non-augmented dataloaders
        train_dl, eval_dl, data, labels = _get_train_eval_loader(args, data_path, flatten_data, resize_shape)

    test_dataset_subset = "test"
    if args.dataset_subset == "all":
        test_dataset_subset = "all"
    # Test loader is always without augmentations
    _, test_dl, test_data, test_labels = _get_train_eval_loader(
        args, data_path, flatten_data, resize_shape, dataset_subset=test_dataset_subset
    )

    # Assign correct dataloaders for AE training
    if args.augmented_pretraining:
        ae_train_dl, ae_eval_dl = aug_train_dl, aug_eval_dl
    else:
        ae_train_dl, ae_eval_dl = train_dl, eval_dl

    # Assign correct dataloaders for Deep Clustering
    if args.augmentation_invariance:
        dc_train_dl, dc_eval_dl = aug_train_dl, aug_eval_dl
    else:
        dc_train_dl, dc_eval_dl = train_dl, eval_dl

    return ae_train_dl, ae_eval_dl, dc_train_dl, dc_eval_dl, test_dl, data, labels, test_data, test_labels
