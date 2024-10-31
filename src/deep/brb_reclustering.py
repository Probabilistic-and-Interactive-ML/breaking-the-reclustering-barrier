import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.brb_config import BRBArgs

from ._clustering_utils import (
    apply_kmeans,
    calculate_supervised_centers,
    embedded_kmeans_prediction,
    encode_batchwise,
    get_nclusters,
    run_clustering,
)
from .soft_reset import soft_reset


def adam_reset_centers_momentum(optimizer: torch.optim.Adam, centers: torch.Tensor, reset_momentum_counts: bool):
    """Reset accumulated momentum parameters of Adam for the cluster centers.
    IMPORTANT: if infer_center_index is used, we assume that the cluster centers where added last to the
    optimizer and have the highest index.

    Parameters
    -----------
    optimizer: torch.optim.Adam
        Adam optimizer that will be reset
    centers: torch.Tensor
        centers for which the momentum will be reset. Only used to check if shapes of centers and optimizer parameters match
    infer_center_index: bool
        Infers the index of the centers in optimizer.state_dict()['state'].
        IMPORTANT: We assume that the cluster centers where added last to the optimizer and have the highest index. (Default: True)
    center_index: int
        index in optimizer.state_dict()['state'][center_index] to access momentum terms of centers

    Returns
    -----------
    optimizer: torch.optim.Adam
        Reset Adam optimizer

    Raises
    -----------
    ValueError:
        center_index is None, but infer_center_index is set to False
    AssertionError:
        center shape is different to optimizer.state_dict()["state"][center_index][key].shape
    """

    # Reset momentum for centers by directly fetching the optimizer state
    for key, value in optimizer.state[centers].items():
        if key != "step":
            optimizer_param = optimizer.state[centers][key]
            # Sanity check
            assert optimizer_param.shape == centers.shape
            optimizer.state[centers][key] = torch.zeros_like(value).to(value.device)

    # Reset the counts
    if reset_momentum_counts:
        optimizer.state[centers]["step"].zero_()

    return optimizer


def update_centers_with_labels(
    n_clusters: int,
    autoencoder: torch.nn.Module,
    labels: np.array,
    dataloader: torch.utils.data.DataLoader,
    embeddings: np.ndarray,
    device: torch.device,
):
    """Calculate new centroids for the data."""
    if get_nclusters(labels) < n_clusters:
        # cluster(s) have been lost, need to recluster
        _, _new_labels = apply_kmeans(embeddings, n_clusters)
        new_centers = calculate_supervised_centers(
            labels=_new_labels, module=autoencoder, dataloader=dataloader, device=device
        )
    else:
        new_centers = calculate_supervised_centers(labels=labels, module=autoencoder, dataloader=dataloader, device=device)

    return new_centers


def brb_short_printout(reset_args: dict[str, int | float | bool]) -> None:
    """Print a single-line message showing what type of modifications in BRB are enabled."""

    if not (
        reset_args["reset_weights"]
        or reset_args["recluster"]
        or reset_args["recalculate_centers"]
        or reset_args["reset_momentum"]
    ):
        print("NO INTERVENTION APPLIED")
    elif reset_args["reset_weights"] and reset_args["recluster"] and reset_args["reset_momentum"]:
        print("APPLYING BRB")
    elif reset_args["reset_weights"] and reset_args["recluster"] and not reset_args["reset_momentum"]:
        print("APPLYING BRB (NO MOMENTUM RESET)")
    elif reset_args["reset_weights"] and not reset_args["recluster"] and reset_args["reset_momentum"]:
        print("APPLYING BRB (NO RECLUSTER)")
    elif (
        reset_args["reset_weights"]
        and not reset_args["recluster"]
        and not reset_args["reset_momentum"]
        and not reset_args["recalculate_centers"]
    ):
        print("APPLYING WEIGHT RESET")
    elif (
        reset_args["reset_weights"]
        and not reset_args["recluster"]
        and not reset_args["reset_momentum"]
        and reset_args["recalculate_centers"]
    ):
        print("APPLYING WEIGHT RESET (WITH CENTROID RECALCULATION)")
    elif not reset_args["reset_weights"] and reset_args["recluster"] and not reset_args["reset_momentum"]:
        print("APPLYING RECLUSTERING")
    elif not reset_args["reset_weights"] and reset_args["recluster"] and reset_args["reset_momentum"]:
        print("APPLYING RECLUSTERING (WITH MOMENTUM RESET)")


def brb_settings_printout(reset_args: dict[str, int | float | bool]) -> None:
    """Print a detailed message showing the full BRB setting."""

    print(pd.DataFrame(reset_args, index=["Value"]).T)


def apply_brb(
    cluster_algorithm: nn.Module,
    autoencoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    reset_args: BRBArgs,
    reset_momentum: bool,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, bool]:
    """Reclustering module for centroid-based clustering methods.

    Performs the following steps:
    1. Soft-reset of the autoencoder.
    2. Re-clustering to generate new centroids.
    3. Reset momentum for centers if adam optimizer is used.

    """

    # Print an informative message about the reset settings
    brb_short_printout(vars(reset_args))

    n_clusters = cluster_algorithm.n_clusters

    # Calculate labels before reset for center recalculation
    if reset_args.recalculate_centers:
        embeddings = encode_batchwise(module=autoencoder, dataloader=dataloader, device=device)
        labels = embedded_kmeans_prediction(
            dataloader=dataloader,
            cluster_centers=cluster_algorithm.centers.detach().cpu().numpy(),
            module=autoencoder,
            embedded_data=embeddings,
        )

    # WEIGHT RESET
    if reset_args.reset_weights:
        soft_reset(
            autoencoder=autoencoder,
            reset_interpolation_factor=reset_args.reset_interpolation_factor,
            reset_interpolation_factor_step=reset_args.reset_interpolation_factor_step,
            reset_batchnorm=reset_args.reset_batchnorm,
            reset_embedding=reset_args.reset_embedding,
            reset_projector=reset_args.reset_projector,
            reset_convlayers=reset_args.reset_convlayers,
        )

    # RECLUSTERING
    if reset_args.recluster:
        # NOTE: Must pass None to reclustering because we want to use the a new embedding after a reset
        new_centroids = run_clustering(
            method=reset_args.reclustering_method,
            n_clusters=n_clusters,
            autoencoder=autoencoder,
            dataloader=dataloader,
            device=device,
            embedded=None,
            subsample_size=reset_args.subsample_size,
        )
        cluster_algorithm.set_centers(new_centroids)

    # RECALCULATE CENTERS
    if reset_args.recalculate_centers:
        if reset_args.recluster:
            warnings.warn(
                "Reclustering and recalculating centers are both enabled. Recalculating centers will be ignored.", UserWarning
            )
        else:
            new_centroids = update_centers_with_labels(
                n_clusters=cluster_algorithm.n_clusters,
                labels=labels,
                autoencoder=autoencoder,
                dataloader=dataloader,
                embeddings=embeddings,
                device=device,
            )
            # Update algorithm centroids
            cluster_algorithm.set_centers(new_centroids)

    # MOMENTUM RESET
    if isinstance(optimizer, torch.optim.Adam) and reset_momentum:
        optimizer = adam_reset_centers_momentum(
            optimizer=optimizer, centers=cluster_algorithm.centers, reset_momentum_counts=reset_args.reset_momentum_counts
        )

    return autoencoder, optimizer, True
