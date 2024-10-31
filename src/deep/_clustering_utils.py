import numpy as np
import torch
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import row_norms
from sklearn_extra.cluster import KMedoids

from config.types import ReClusteringArg

from ._torch_utils import detect_device


def calculate_supervised_centers(labels, module, dataloader, device):
    """Embedds data from dataloader with module and calculates means w.r.t. labels"""
    embedded_data = encode_batchwise(dataloader, module, device)
    labels_np = np.array(labels)
    # get unique labels in ascending order
    unique_labels = list(set(labels))
    unique_labels.sort()
    new_centers = []
    for label_i in unique_labels:
        center_i = embedded_data[labels_np == label_i].mean(axis=0).reshape(1, -1)
        new_centers.append(center_i)
    new_centers = np.concatenate(new_centers)
    return new_centers


def squared_euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the pairwise squared euclidean distance between two tensors.
    Each row in the tensors is interpreted as a separate object, while each column represents its features.
    Therefore, the result of an (4x3) and (12x3) tensor will be a (4x12) tensor.
    Optionally, features can be individually weighted.
    The default behavior is that all features are weighted by 1.

    Parameters
    ----------
    tensor1 : torch.Tensor
        the first tensor
    tensor2 : torch.Tensor
        the second tensor
    weights : torch.Tensor
        tensor containing the weights of the features (default: None)

    Returns
    -------
    squared_diffs : torch.Tensor
        the pairwise squared euclidean distances
    """
    assert tensor1.shape[1] == tensor2.shape[1], "The number of features of the two input tensors must match."
    ta = tensor1.unsqueeze(1)
    tb = tensor2.unsqueeze(0)
    squared_diffs = ta - tb
    if weights is not None:
        assert tensor1.shape[1] == weights.shape[0]
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).sum(2)
    return squared_diffs


def predict_batchwise(
    dataloader: torch.utils.data.DataLoader, module: torch.nn.Module, cluster_module: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """
    Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion.
    Method calls the predict_hard method of the cluster_module for each batch of data.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    module : torch.nn.Module
        the module that is used for the encoding (e.g. an autoencoder)
    cluster_module : torch.nn.Module
        the cluster module that is used for the encoding (e.g. DEC). Usually contains the predict method.
    device : torch.device
        device to be trained on

    Returns
    -------
    predictions_numpy : np.ndarray
        The predictions of the cluster_module for the data set
    """
    predictions = []
    for batch in dataloader:
        batch_data = batch[1].to(device)
        prediction = cluster_module.predict_hard(module.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    predictions_numpy = torch.cat(predictions, dim=0).numpy()
    return predictions_numpy


def apply_kmeans(data: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns KMeans cluster_centers and cluster labels"""
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


@torch.inference_mode()
def encode_batchwise(
    dataloader: torch.utils.data.DataLoader, module: torch.nn.Module, device: torch.device, subsample_size: int = None
) -> np.ndarray:
    """
    Utility function for embedding the whole data set in a mini-batch fashion

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    module : torch.nn.Module
        the module that is used for the encoding (e.g. an autoencoder)
    device : torch.device
        device to be trained on
    subsample_size: int | None
        size of subsample that should be encoded (stops mini-batch encoding once subsample_size is reached).
        If None, then all data will be encoded (default: None)

    Returns
    -------
    embeddings_numpy : np.ndarray
        The embedded data set
    """
    embeddings = []
    sample_size_i = 0
    if subsample_size is None:
        subsample_size = np.inf
    for batch in dataloader:
        batch_data = batch[1].to(device)
        if sample_size_i < subsample_size:
            emb_i = module.encode(batch_data)
        else:
            break
        embeddings.append(emb_i.detach().cpu())
        sample_size_i += batch_data.shape[0]
    embeddings_numpy = torch.cat(embeddings, dim=0).numpy()
    # remove surplus samples, e.g., because last added batch increased size over subsample size
    if embeddings_numpy.shape[0] > subsample_size:
        embeddings_numpy = embeddings_numpy[:subsample_size]
    return embeddings_numpy


def embedded_kmeans_prediction(
    dataloader: torch.utils.data.DataLoader,
    cluster_centers: np.ndarray,
    module: torch.nn.Module,
    device=None,
    embedded_data=None,
) -> np.array:
    """
    Predicts the labels of the data within the given dataloader.
    Labels correspond to the id of the closest cluster center.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    cluster_centers : np.ndarray
        input cluster centers
    module : torch.nn.Module
        the module that is used for the encoding (e.g. an autoencoder)
    device : torch.device
        Device on which prediction is done. If none then available cuda or cpu device will be used. (default: None)
    embedded_data : np.ndarray
        embedded data (default: None)
    Returns
    -------
    predicted_labels : np.array
        The predicted labels
    """
    if device is None:
        device = detect_device()
    if embedded_data is None:
        embedded_data = encode_batchwise(dataloader, module, device)
    predicted_labels, _ = pairwise_distances_argmin_min(
        X=embedded_data, Y=cluster_centers, metric="euclidean", metric_kwargs={"squared": True}
    )
    return predicted_labels


def get_nclusters(labels: np.array) -> int:
    return len(set(labels.tolist()))


def _kmeans_plus_plus(
    X: np.ndarray,
    n_clusters: int,
    x_squared_norms: np.ndarray,
    n_local_trials: int = None,
):
    """
    Initializes the cluster centers for Kmeans using the kmeans++ procedure.
    This method is only a wrapper for the Sklearn kmeans++ implementation.
    Output and naming of kmeans++ in Sklearn changed multiple times. This wrapper can work with multiple versions.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        list containing number of clusters for each subspace
    x_squared_norms : np.ndarray
        Row-wise (squared) Euclidean norm of X. See sklearn.utils.extmath.row_norms
    n_local_trials : int
        Number of local trials (default: None)

    Returns
    -------
    centers : np.ndarray
        The resulting initial cluster centers.

    References
    ----------
    Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful seeding".
    ACM-SIAM symposium on Discrete algorithms. 2007
    """
    centers = kmeans_plusplus(X, n_clusters, x_squared_norms=x_squared_norms, n_local_trials=n_local_trials)
    if type(centers) is tuple:
        centers = centers[0]
    return centers


def run_clustering(
    method: ReClusteringArg,
    n_clusters: int,
    autoencoder: torch.nn.Module | None,
    dataloader: torch.utils.data.DataLoader | None,
    device: torch.device,
    embedded: np.ndarray | None = None,
    subsample_size: int | None = None,
) -> np.ndarray:
    if embedded is None:
        embedded = encode_batchwise(dataloader, autoencoder, device, subsample_size=subsample_size)
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(embedded)
        centers = kmeans.cluster_centers_
    elif method == "kmeans++-init":
        centers = _kmeans_plus_plus(embedded, n_clusters, row_norms(embedded, squared=True))
    elif method == "kmedoids":
        kmedoids = KMedoids(n_clusters=n_clusters)
        kmedoids.fit(embedded)
        centers = kmedoids.cluster_centers_
    elif method == "em":
        mixture = GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5)
        mixture.fit(embedded)
        # Ensure centers are float32 for DEC/IDEC backprop
        centers = mixture.means_.astype(np.float32)
    elif method == "random":
        rnd_indices = np.random.randint(low=0, high=embedded.shape[0], size=n_clusters)
        centers = embedded[rnd_indices]
    else:
        raise ValueError(f"Method {method} is not implemented.")

    return centers
