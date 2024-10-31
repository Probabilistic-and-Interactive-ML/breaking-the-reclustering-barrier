import ssl

import numpy as np
import torch
import torchvision

from .data_utils import _get_download_dir, _load_data_file, _load_image_data

"""
Load torchvision datasets
"""


def flatten_images(data: np.ndarray, format: str) -> np.ndarray:
    """
    Convert data array from image to numerical vector.
    Before flattening, color images will be converted to the HWC/HWDC (height, width, color channels) format.

    Parameters
    ----------
    data : np.ndarray
        The given data set
    format : str
        Format of the images with the data array. Can be: "HW", "HWD", "CHW", "CHWD", "HWC", "HWDC".
        Abbreviations stand for: H: Height, W: Width, D: Depth, C: Color-channels

    Returns
    -------
    data : np.ndarray
        The flatten data array
    """
    format_possibilities = ["HW", "HWD", "CHW", "CHWD", "HWC", "HWDC"]
    assert format in format_possibilities, f"Format must be within {format_possibilities}"
    if format == "HW":
        assert data.ndim == 3
    elif format in ["HWD", "CHW", "HWC"]:
        assert data.ndim == 4
    elif format in ["CHWD", "HWDC"]:
        assert data.ndim == 5
    # Flatten shape
    if format != "HW" and format != "HWD":
        if format == "CHW":
            # Change representation to HWC
            data = np.transpose(data, [0, 2, 3, 1])
        elif format == "CHWD":
            # Change representation to HWDC
            data = np.transpose(data, [0, 2, 3, 4, 1])
        assert (
            data.shape[-1] == 3
        ), f"Color-channels must be in the last position and contain three channels not {data.shape[-1]} ({data.shape})"
    data = data.reshape(data.shape[0], -1)
    return data


def _get_data_and_labels(dataset: torchvision.datasets.VisionDataset, image_size: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract data and labels from a torchvision dataset object.

    Parameters
    ----------
    dataset : torchvision.datasets.VisionDataset
        The torchvision dataset object
    image_size : tuple
        for some datasets (e.g., GTSRB) the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images

    Returns
    -------
    data, labels : (torch.Tensor, torch.Tensor)
        the data torch tensor, the labels torch tensor
    """
    if hasattr(dataset, "data"):
        # USPS, MNIST, ... use data parameter
        data = dataset.data
        if hasattr(dataset, "targets"):
            # USPS, MNIST, ... use targets
            labels = dataset.targets
        else:
            labels = dataset.labels
    else:
        # GTSRB only gives path to images
        labels = []
        data_list = []
        for path, label in dataset._samples:
            labels.append(label)
            image_data = _load_image_data(path, image_size, True)
            data_list.append(image_data)
        # Convert data form list to numpy array
        data = np.array(data_list)
        labels = np.array(labels)
    if type(data) is np.ndarray:
        # Transform numpy arrays to torch tensors. Needs to be done for eg USPS
        data = torch.from_numpy(data)
        labels = torch.from_numpy(np.array(labels))
    return data, labels


def _load_torch_image_data(
    data_source: torchvision.datasets.VisionDataset,
    subset: str,
    flatten: bool,
    normalize_channels: bool,
    uses_train_param: bool,
    downloads_path: str,
    is_color_channel_last: bool,
    image_size: tuple = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to load a data set from the torchvision package.
    All data sets will be returned as a two-dimensional tensor, created out of the HWC (height, width, color channels) image representation.

    Parameters
    ----------
    data_source : torchvision.datasets.VisionDataset
        the data source from torchvision.datasets
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, color images will be returned in the CHW format
    normalize_channels : bool
        normalize each color-channel of the images
    uses_train_param : bool
        is the test/train parameter called 'train' or 'split' in the data loader. uses_train_param = True corresponds to 'train'
    downloads_path : str
        path to the directory where the data is stored
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images
    image_size : tuple
        for some datasets (e.g., GTSRB) the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images (default: None)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    subset = subset.lower()
    assert subset in [
        "all",
        "train",
        "test",
        "train+unlabeled",
    ], f"subset must match 'all', 'train' or 'test'. Your input {subset}"
    # Get data from source
    default_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    if subset == "all" or subset == "train":
        # Load training data
        if uses_train_param:
            trainset = data_source(root=_get_download_dir(downloads_path), train=True, download=True)
        else:
            trainset = data_source(root=_get_download_dir(downloads_path), split="train", download=True)
        data, labels = _get_data_and_labels(trainset, image_size)
    elif subset == "train+unlabeled":
        # Load training data
        trainset = data_source(root=_get_download_dir(downloads_path), split="train+unlabeled", download=True)
        data, labels = _get_data_and_labels(trainset, image_size)
    if subset == "all" or subset == "test":
        # Load test data
        if uses_train_param:
            testset = data_source(root=_get_download_dir(downloads_path), train=False, download=True)
        else:
            testset = data_source(root=_get_download_dir(downloads_path), split="test", download=True)
        data_test, labels_test = _get_data_and_labels(testset, image_size)
        if subset == "all":
            # Add to train data
            data = torch.cat([data, data_test], dim=0)
            labels = torch.cat([labels, labels_test], dim=0)
        else:
            data = data_test
            labels = labels_test
    # Convert data to float and labels to int
    data = data.float()
    labels = labels.int()
    ssl._create_default_https_context = default_ssl
    # Check data dimensions
    if data.dim() < 3 or data.dim() > 5:
        raise Exception(f"Number of dimensions for torchvision data sets should be 3, 4 or 5. Here dim={data.dim()}")
    # Normalize and flatten
    data = _torch_normalize_and_flatten(data, flatten, normalize_channels, is_color_channel_last)
    # Move data to CPU
    data_cpu = data.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return data_cpu, labels_cpu


def _torch_normalize_and_flatten(data: torch.Tensor, flatten: bool, normalize_channels: bool, is_color_channel_last: bool):
    """
    Helper function to load a data set from the torchvision package.
    All data sets will be returned as a two-dimensional tensor, created out of the HWC (height, width, color channels) image representation.

    Parameters
    ----------
    data : torch.Tensor
        The torch data tensor
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, color images will be returned in the CHW format
    normalize_channels : bool
        normalize each color-channel of the images
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images

    Returns
    -------
    data : torch.Tensor
        The (non-)normalized and (non-)flatten data tensor
    """
    # Channels can be normalized
    if normalize_channels:
        data = _torch_normalize_channels(data, is_color_channel_last)
    # Flatten shape
    if flatten:
        data = _torch_flatten_shape(data, is_color_channel_last, normalize_channels)
    elif not normalize_channels:
        # Change image to CHW format
        if data.dim() == 4 and is_color_channel_last:  # equals 2d color images
            # Change to CHW representation
            data = data.permute(0, 3, 1, 2)
            assert data.shape[1] == 3, "Colored image must consist of three channels not " + data.shape[1]
        elif data.dim() == 5 and is_color_channel_last:  # equals 3d color-images
            # Change to CHWD representation
            data = data.permute(0, 4, 1, 2, 3)
            assert data.shape[1] == 3, f"Colored image must consist of three channels not {data.shape[1]}"
    return data


def _torch_normalize_channels(data: torch.Tensor, is_color_channel_last: bool) -> torch.Tensor:
    """
    Normalize the color channels of a torch dataset

    Parameters
    ----------
    data : torch.Tensor
        The torch data tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images

    Returns
    -------
    data : torch.Tensor
        The normalized data tensor in CHW format
    """
    if data.dim() == 3 or (data.dim() == 4 and is_color_channel_last is None):
        # grayscale images (2d or 3d)
        data_mean = [data.mean()]
        data_std = [data.std()]
    elif data.dim() == 4:  # equals 2d color images
        if is_color_channel_last:
            # Change to CHW representation
            data = data.permute(0, 3, 1, 2)
        assert data.shape[1] == 3, "Colored image must consist of three channels not " + data.shape[1]
        # color images
        data_mean = data.mean([0, 2, 3])
        data_std = data.std([0, 2, 3])
    elif data.dim() == 5:  # equals 3d color-images
        if is_color_channel_last:
            # Change to CHWD representation
            data = data.permute(0, 4, 1, 2, 3)
        assert data.shape[1] == 3, f"Colored image must consist of three channels not {data.shape[1]}"
        # color images
        data_mean = data.mean([0, 2, 3, 4])
        data_std = data.std([0, 2, 3, 4])
    normalize = torchvision.transforms.Normalize(data_mean, data_std)
    data = normalize(data)
    return data


def _torch_flatten_shape(data: torch.Tensor, is_color_channel_last: bool, normalize_channels: bool) -> torch.Tensor:
    """
    Convert torch data tensor from image to numerical vector.

    Parameters
    ----------
    data : torch.Tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images
    normalize_channels : bool
        normalize each color-channel of the images

    Returns
    -------
    data : torch.Tensor
        The flatten data vector
    """
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    elif data.dim() == 4:
        # In case of 3d grayscale image is_color_channel_last is None
        if is_color_channel_last is not None and (not is_color_channel_last or normalize_channels):
            # Change representation to HWC
            data = data.permute(0, 2, 3, 1)
        assert (
            is_color_channel_last is None or data.shape[3] == 3
        ), f"Colored image must consist of three channels not {data.shape[3]}"
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    elif data.dim() == 5:
        if not is_color_channel_last or normalize_channels:
            # Change representation to HWDC
            data = data.permute(0, 2, 3, 4, 1)
        assert data.shape[4] == 3, f"Colored image must consist of three channels not {data.shape[4]}"
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4])
    return data


def load_mnist(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST data set. It consists of 70000 28x28 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : bool
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.MNIST, subset, flatten, normalize_channels, True, downloads_path, None
    )
    return data, labels


def load_kmnist(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Kuzushiji-MNIST data set. It consists of 70000 28x28 grayscale images showing Kanji characters.
    It is composed of 10 different characters, each representing one column of hiragana.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.KMNIST, subset, flatten, normalize_channels, True, downloads_path, None
    )
    return data, labels


def load_fmnist(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Fashion-MNIST data set. It consists of 70000 28x28 grayscale images showing articles from the Zalando online store.
    Each sample belongs to one of 10 product groups.
    The data set is composed of 60000 training and 10000 test images.
    N=70000, d=784, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (70000 x 784), the labels numpy array (70000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.FashionMNIST, subset, flatten, normalize_channels, True, downloads_path, None
    )
    return data, labels


def load_optdigits(subset: str = "all", flatten: bool = True, downloads_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the optdigits data set. It consists of 5620 8x8 grayscale images, each representing a digit (0 to 9).
    Each pixel depicts the number of marked pixel within a 4x4 block of the original 32x32 bitmaps.
    The data set is composed of 3823 training and 1797 test samples.
    N=5620, d=64, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (5620 x 64), the labels numpy array (5620)

    References
    -------
    http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
    """
    subset = subset.lower()
    assert subset in ["all", "train", "test"], f"subset must match 'all', 'train' or 'test'. Your input {subset}"
    if subset == "all" or subset == "train":
        filename = _get_download_dir(downloads_path) + "/optdigits.tra"
        data, labels = _load_data_file(
            filename, "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
        )
    if subset == "all" or subset == "test":
        filename = _get_download_dir(downloads_path) + "/optdigits.tes"
        test_data, test_labels = _load_data_file(
            filename, "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
        )
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    if not flatten:
        data = data.reshape((-1, 8, 8))
    return data, labels


def load_usps(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the USPS data set. It consists of 9298 16x16 grayscale images showing handwritten digits (0 to 9).
    The data set is composed of 7291 training and 2007 test images.
    N=9298, d=256, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (9298 x 256), the labels numpy array (9298)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html#torchvision.datasets.USPS
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.USPS, subset, flatten, normalize_channels, True, downloads_path, None
    )
    return data, labels


def load_cifar10(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the CIFAR10 data set. It consists of 60000 32x32 color images showing different objects.
    The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=10.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.CIFAR10, subset, flatten, normalize_channels, True, downloads_path, True
    )
    return data, labels


def load_cifar100_20(
    subset: str = "all", flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the CIFAR100-20 data set. It consists of 60000 32x32 color images showing different objects.
    In general, it is equal to the CIFAR100 data set but only the 20 superclasses are used.
    The data set is composed of 50000 training and 10000 test images.
    N=60000, d=3072, k=20.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (60000 x 3072), the labels numpy array (60000)

    References
    -------
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.CIFAR100, subset, flatten, normalize_channels, True, downloads_path, True
    )
    class_obj_to_ids = {
        "apple": 0,
        "aquarium_fish": 1,
        "baby": 2,
        "bear": 3,
        "beaver": 4,
        "bed": 5,
        "bee": 6,
        "beetle": 7,
        "bicycle": 8,
        "bottle": 9,
        "bowl": 10,
        "boy": 11,
        "bridge": 12,
        "bus": 13,
        "butterfly": 14,
        "camel": 15,
        "can": 16,
        "castle": 17,
        "caterpillar": 18,
        "cattle": 19,
        "chair": 20,
        "chimpanzee": 21,
        "clock": 22,
        "cloud": 23,
        "cockroach": 24,
        "couch": 25,
        "crab": 26,
        "crocodile": 27,
        "cup": 28,
        "dinosaur": 29,
        "dolphin": 30,
        "elephant": 31,
        "flatfish": 32,
        "forest": 33,
        "fox": 34,
        "girl": 35,
        "hamster": 36,
        "house": 37,
        "kangaroo": 38,
        "keyboard": 39,
        "lamp": 40,
        "lawn_mower": 41,
        "leopard": 42,
        "lion": 43,
        "lizard": 44,
        "lobster": 45,
        "man": 46,
        "maple_tree": 47,
        "motorcycle": 48,
        "mountain": 49,
        "mouse": 50,
        "mushroom": 51,
        "oak_tree": 52,
        "orange": 53,
        "orchid": 54,
        "otter": 55,
        "palm_tree": 56,
        "pear": 57,
        "pickup_truck": 58,
        "pine_tree": 59,
        "plain": 60,
        "plate": 61,
        "poppy": 62,
        "porcupine": 63,
        "possum": 64,
        "rabbit": 65,
        "raccoon": 66,
        "ray": 67,
        "road": 68,
        "rocket": 69,
        "rose": 70,
        "sea": 71,
        "seal": 72,
        "shark": 73,
        "shrew": 74,
        "skunk": 75,
        "skyscraper": 76,
        "snail": 77,
        "snake": 78,
        "spider": 79,
        "squirrel": 80,
        "streetcar": 81,
        "sunflower": 82,
        "sweet_pepper": 83,
        "table": 84,
        "tank": 85,
        "telephone": 86,
        "television": 87,
        "tiger": 88,
        "tractor": 89,
        "train": 90,
        "trout": 91,
        "tulip": 92,
        "turtle": 93,
        "wardrobe": 94,
        "whale": 95,
        "willow_tree": 96,
        "wolf": 97,
        "woman": 98,
        "worm": 99,
    }
    new_labels = {
        0: ["beaver", "dolphin", "otter", "seal", "whale"],
        1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
        3: ["bottle", "bowl", "can", "cup", "plate"],
        4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        5: ["clock", "keyboard", "lamp", "telephone", "television"],
        6: ["bed", "chair", "couch", "table", "wardrobe"],
        7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        8: ["bear", "leopard", "lion", "tiger", "wolf"],
        9: ["bridge", "castle", "house", "road", "skyscraper"],
        10: ["cloud", "forest", "mountain", "plain", "sea"],
        11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
        13: ["crab", "lobster", "snail", "spider", "worm"],
        14: ["baby", "boy", "girl", "man", "woman"],
        15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }
    labels_new = np.zeros(labels.shape, dtype=np.int32) - 1
    for nl in new_labels.keys():
        for object_name in new_labels[nl]:
            labels_new[labels == class_obj_to_ids[object_name]] = nl
    return data, labels_new


def load_gtsrb(
    subset: str = "all",
    image_size: tuple = (32, 32),
    flatten: bool = True,
    normalize_channels: bool = False,
    downloads_path: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the GTSRB (German Traffic Sign Recognition Benchmark) data set. It consists of 39270 color images showing 43 different traffic signs.
    Example classes are: stop sign, speed limit 50 sign, speed limit 70 sign, construction site sign and many others.
    The data set is composed of 26640 training and 12630 test images.
    N=39270, d=image_size[0]*image_size[1]*3, k=43.

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    image_size : tuple
        the images of various sizes must be converted into a coherent size.
        The tuple equals (width, height) of the images (default: (32, 32))
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)


    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (39270 x image_size[0]*image_size[1]*3), the labels numpy array (20580)

    References
    -------
    https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html#torchvision.datasets.GTSRB

    and

    https://benchmark.ini.rub.de/
    """
    data, labels = _load_torch_image_data(
        torchvision.datasets.GTSRB, subset, flatten, normalize_channels, False, downloads_path, True, image_size
    )
    return data, labels
