import os
import ssl
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_datafiles")


def _get_download_dir(downloads_path: str) -> str:
    """
    Helper function to define the path where the data files should be stored. If downloads_path is None then default path
    '[USER]/Downloads/clustpy_datafiles' will be used. If the directory does not exists it will be created.

    Parameters
    ----------
    downloads_path : str
        path to the directory where the data will be stored. Can be None

    Returns
    -------
    downloads_path : str
        path to the directory where the data will be stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'
    """
    if downloads_path is None:
        env_data_path = os.environ.get("CLUSTPY_DATA", None)
        if env_data_path is None:
            downloads_path = DEFAULT_DOWNLOAD_PATH
        else:
            downloads_path = env_data_path
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
        with open(downloads_path + "/info.txt", "w") as f:
            f.write(
                "This directory was created by the ClustPy python package to store real world data sets.\n"
                "The default directory is '[USER]/Downloads/clustpy_datafiles' and can be changed with the "
                "'downloads_path' parameter when loading a data set.\n"
                "Alternatively, a global python environment variable for the path can be defined with os.environ['CLUSTPY_DATA'] = 'PATH'."
            )
    return downloads_path


def _download_file(file_url: str, filename_local: str) -> None:
    """
    Helper function to download a file into a specified location.

    Parameters
    ----------
    file_url : str
        URL of the file
    filename_local : str
        local name of the file after it has been downloaded
    """
    print(f"Downloading data set from {0} to {1}".format(file_url, filename_local))
    default_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(file_url, filename_local)
    ssl._create_default_https_context = default_ssl


def _load_data_file(
    filename_local: str, file_url: str, delimiter: str = ",", last_column_are_labels: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to load a data file. Either the first or last column, depending on last_column_are_labels, of the
    data file is used as the label column.
    If file does not exist on the local machine it will be downloaded.

    Parameters
    ----------
    filename_local : str
        local name of the file after it has been downloaded
    file_url : str
        URL of the file
    delimiter : str
        delimiter in the data file (default: ";")
    last_column_are_labels : bool
        specifies if the last column contains the labels. If false labels should be contained in the first column (default: True)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    if not os.path.isfile(filename_local):
        _download_file(file_url, filename_local)
    datafile = np.genfromtxt(filename_local, delimiter=delimiter)
    if last_column_are_labels:
        data = datafile[:, :-1]
        labels = datafile[:, -1]
    else:
        data = datafile[:, 1:]
        labels = datafile[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    return data, labels


def _load_image_data(path: str, image_size: tuple, color_image: bool) -> np.ndarray:
    """
    Load image and convert it into a coherent size. Returns a numpy array containing the image data.

    Parameters
    ----------
    path : str
        Path to the image
    image_size : tuple
        images of various sizes can be converted into a coherent size.
        The tuple equals (width, height) of the images.
        Can also be None if the image size should not be changed
    color_image : bool
        Specifies if the loaded image is a color image

    Returns
    -------
    image_data : np.ndarray
        The numpy array containing the image data
    """
    image = Image.open(path)
    if color_image:
        image = image.convert("RGB")
    # Convert to coherent size
    if image_size is not None:
        image = image.resize(image_size)
    image_data = np.asarray(image)
    assert image_size is None or image_data.shape == (
        image_size[0],
        image_size[1],
        3,
    ), f"Size of image is not correct. Should be {image_size} but is {image_data.shape}."
    return image_data
