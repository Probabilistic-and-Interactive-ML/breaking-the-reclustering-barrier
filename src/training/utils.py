import os
import typing
from io import BytesIO
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pynvml.smi import nvidia_smi
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA

from config.base_config import Args, DataSetArg


def set_cuda_configuration(gpu: typing.Any) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""
    if gpu is None or gpu == -1 or gpu is False:
        device = torch.device("cpu")
    elif isinstance(gpu, int):
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")
    else:
        raise ValueError(f"Invalid GPU argument: {gpu}.")

    return device


def get_gpu_with_most_free_memory(gpus: tuple[int, ...] | list[int] | str) -> typing.Optional[int]:
    """Check which GPU currently has the lowest usage and return its ID."""

    if len(gpus) == 1:
        # Only a single GPU is specified -> Use that one
        to_use = gpus[0]
    else:
        # Sanity check
        assert isinstance(gpus, tuple | list | str), f"gpus={gpus} of type={type(gpus)} is not supported."

        # Wandb sweeps do not properly pass the arguments, so we need to parse the string manually
        if gpus == "all":
            gpus = [i for i in range(torch.cuda.device_count())]
        elif isinstance(gpus, str) and gpus.startswith("[") and gpus.endswith("]"):
            gpus = [int(gpu) for gpu in gpus.lstrip("[").rstrip("]").split(",")]

        # Multiple GPUs specified -> Check which one has the least memory usage and set it
        nvsmi = nvidia_smi.getInstance()
        all_stats = nvsmi.DeviceQuery("memory.free, memory.total")["gpu"]
        # Get only the stats of the GPUs that were specified in the config with their correct index
        selected_stats = {i: all_stats[i]["fb_memory_usage"]["free"] for i in set(gpus)}
        # Select the GPU with the maximum free memory available
        to_use = max(selected_stats, key=selected_stats.get)  # type: ignore[arg-type]
        print(f"GPU usage statistics: {selected_stats}.")
        print(f"Scheduling task on GPU: {to_use}.")

    return to_use


def get_number_of_clusters(dataset_name: str, dataset_subset: str) -> int:
    assert dataset_name in get_args(DataSetArg)

    if dataset_name == "gtsrb":
        n_clusters = 5
    elif dataset_name == "cifar100_20":
        n_clusters = 20
    else:
        n_clusters = 10
    return n_clusters


def get_ae_path(args: Args, data_path: Path) -> Path:
    if args.ae_path is not None:
        return args.ae_path
    else:
        # Set default values for saving to make loading easier
        if not args.use_contrastive_loss:
            softmax_temperature_tau = 0.0
            projector_depth = 0
            projector_layer_size = None
            separate_cluster_head = False
        else:
            softmax_temperature_tau = args.softmax_temperature_tau
            projector_depth = args.projector_depth
            projector_layer_size = args.projector_layer_size
            separate_cluster_head = args.separate_cluster_head

        if not args.augmented_pretraining:
            augment_both_views = False
        else:
            augment_both_views = args.augment_both_views

        ae_path = Path(f"{data_path}/models/{args.dataset_name}/{args.dataset_subset}")

        ae_name = (
            f"{args.seed}_"
            f"{args.model_type}_"
            f"{args.convnet}_"
            f"{args.pretrain_optimizer.weight_decay}_"
            f"{args.pretrain_optimizer.lr}_"
            f"{args.pretrain_optimizer.schedule_lr}_"
            f"{args.pretrain_epochs}_"
            f"{args.activation_fn}_"
            f"{args.batch_norm}_"
            f"{args.pretrain_optimizer.optimizer}_"
            f"{args.embedding_dim}_"
            f"{args.augmented_pretraining}_"
            f"{augment_both_views}_"
            f"{args.use_contrastive_loss}_"
            f"{softmax_temperature_tau}_"
            f"{projector_depth}_"
            f"{projector_layer_size}_"
            f"{separate_cluster_head}_"
            f"{args.crop_size}_"
            "ae.pt"
        )

        return ae_path / ae_name


def pca_voronoi_plot(points: np.array, labels: np.array, folder_path: str = None) -> np.array:
    """
    Performs PCA on the points, recalculates centroids, creates a Voronoi plot, and saves it.

    :param points: ndarray of shape (n_samples, n_features)
    :param labels: ndarray of shape (n_samples,)
    :param folder_path: Path to the folder where the plot will be saved
    """

    if folder_path is not None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)

    unique_labels = np.unique(labels)
    centroids = np.array([points_2d[labels == label].mean(axis=0) for label in unique_labels])

    vor = Voronoi(centroids)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="orange", line_width=2, line_alpha=0.6, point_size=2)

    for label in unique_labels:
        ax.scatter(*points_2d[labels == label].T, label=f"Label {label}")

    ax.legend()

    if folder_path is None:
        # Convert the plot to a PNG in memory
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        # Convert the buffer into a PIL Image, then into a NumPy array
        image = Image.open(buf)
        image_np = np.array(image)
        buf.close()

        return image_np
    save_path = os.path.join(folder_path, "voronoi_plot.png")
    plt.savefig(save_path)
    plt.close(fig)

    return None
