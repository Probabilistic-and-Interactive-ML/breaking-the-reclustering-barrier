import os
import random

import numpy as np
import torch


def detect_device(device: torch.device = None) -> torch.device:
    """
    Automatically detects if you have a cuda enabled GPU.
    Device can also be read from environment variable "CLUSTPY_DEVICE".
    It can be set using, e.g., os.environ["CLUSTPY_DEVICE"] = "cuda:1"

    Parameters
    ----------
    device : torch.device
        the input device. Will be returned if it is not None (default: None)

    Returns
    -------
    device : torch.device
        device on which the prediction should take place
    """
    if device is None:
        env_device = os.environ.get("CLUSTPY_DEVICE", None)
        # Check if environment device is None
        if env_device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(env_device)
    return device


def set_torch_seed(random_state: np.random.RandomState) -> None:
    """
    Set the random state for torch applications.

    Parameters
    ----------
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    """
    seed = (random_state.get_state()[1][0]).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
