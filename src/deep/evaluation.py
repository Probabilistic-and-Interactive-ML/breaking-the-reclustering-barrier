import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.deep._clustering_utils import embedded_kmeans_prediction
from src.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from src.deep.silhouette_score import silhouette_score
from src.deep.uncertainty_score import uncertainty_score
from src.training.utils import pca_voronoi_plot
from wandb import Image as wandb_image


@torch.inference_mode()
def encode_decode_batchwise(
    dataloader: torch.utils.data.DataLoader, module: torch.nn.Module, device: torch.device, loss_fn=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Utility function for encoding and decoding the whole data set in a mini-batch fashion with an autoencoder.
    Note: Assumes an implemented decode function

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to be used
    module : torch.nn.Module
        the module that is used for the encoding and decoding (e.g. an autoencoder)
    device : torch.device
        device to be trained on
    loss_fn : torch.nn.Loss
        loss_fn to be used for calculating difference between input and decoding. If None, rec_loss will not be returned. (Default: None)

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        The embedded data set,
        The reconstructed data set
    """
    embeddings = []
    reconstructions = []
    rec_loss = torch.tensor(0.0, device=device)
    for batch in dataloader:
        batch_data = batch[1].to(device)
        embedding = module.encode(batch_data)
        embeddings.append(embedding.detach().cpu())
        rec = module.decode(embedding)
        if loss_fn is not None:
            rec_loss += loss_fn(batch_data, rec).detach()
        else:
            rec_loss += 0.0
        reconstructions.append(rec.detach().cpu())
    # NOTE: Sync only once at the end because .cpu() is a rather expensive call, same with .item()
    embeddings_numpy = torch.cat(embeddings, dim=0).numpy()
    reconstructions_numpy = torch.cat(reconstructions, dim=0).numpy()
    rec_loss_numpy = rec_loss.item() / len(dataloader)
    if loss_fn is not None:
        return embeddings_numpy, reconstructions_numpy, rec_loss_numpy
    else:
        return embeddings_numpy, reconstructions_numpy, rec_loss_numpy


def compute_nearest_neighbor_indices(X: torch.Tensor, nn: int | None, device: torch.device) -> torch.Tensor:
    """Compute nearest neighbor indices

    Parameters
    ----------
    X : torch.Tensor
        the given data set needed for computing nn_indices.
    nn: int | None
        number of neighbours to consider.
        If None, nn = X.shape[0]
    device: torch.device
        device on which computation should be done.

    Returns
    ----------
    nn_indices: torch.Tensor (N x nn)
        nearest neighbor indices
    """
    X = X.to(device)
    # calculate euclidean distance matrix
    dist = torch.cdist(X, X)
    if nn is None:
        nn = X.shape[0]
    # get nearest neighbors for each instance (including instance itself)
    _, nn_indices = dist.topk(nn, largest=False, dim=1)
    return nn_indices.cpu()


def local_purity_torch(
    X: torch.Tensor | None, y: torch.Tensor, nn: int, precomputed_nn_indices: torch.Tensor | None = None
) -> float:
    """
    Reimplementation of local purity using PyTorch (allows fast computation on GPU).
    Assumes X, y are on same device (cuda | cpu)

    Local purity measures the average label purity in the neighborhood of all instances.
    For example, consider a nearest neighbors (nn) size of 10 for an instance x with label y.
    We now count the occurences of instances x_1, ..., x_10 where the corresponding labels y_1, ..., y_10
    are equal to y. We average this over all occurences and get the purity of a single instance.

    The overall local purity for certain nn size is then the average over all single instance purities.

    In the best case, the local purity is 1, if all neighborhoods contain only instances of a single class.
    In the worst case the local purity is 1/k, where k is the number of clusters.

    Parameters
    ----------
    X : torch.Tensor | None
        the given data set needed for computing nn_indices. Can be None if precomputed_nn_indices are given.
    y: torch.Tensor
        tensor of class labels
    nn: int
        number of neighbours to consider
    precomputed_nn_indices: torch.Tensor | None
        N times nn+1 Matrix of indices used for nearest neighbor lookup (Can be precomputed to save time).
        Each row contains nn+1 indices (the instance itself with its nn nearest neighbors).
        If None, distance matrix from X will be computed and nearest neighbors will be calculated. (default: None)

    Returns
    ----------
    local_purity: float
        scalar value of local purity

    Raises
    ----------
    ValueError
        If precomputed_nn_indices contain {precomputed_nn_indices.shape[1]} less than (nn+1)={nn+1} indices.

    References
    ----------
    Aleksandar Bojchevski, Yves Matkovic, Stephan Günnemann:
    Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings.
    KDD 2017: 737-746
    """
    if precomputed_nn_indices is None:
        nn_indices = compute_nearest_neighbor_indices(X, nn + 1, device=X.device)
    else:
        if precomputed_nn_indices.shape[1] < (nn + 1):
            raise ValueError(
                f"precomputed_nn_indices contain {precomputed_nn_indices.shape[1]} indices, which is less than (nn+1)={nn+1} indices."
            )
        nn_indices = precomputed_nn_indices[:, : nn + 1]
    # unique classes
    y_unique = torch.tensor(list(set(y.tolist())))
    # count for each class how often it occured in the neighborhood of a point
    neighborhood_frequencies = (y[nn_indices].unsqueeze(-1) == y_unique.reshape(1, -1)).sum(1)
    # normalize frequencies by nn+1
    neighborhood_frequencies = neighborhood_frequencies / (nn + 1)
    # looks for most often occuring class in a neighborhood of each instance
    instancewise_purity = neighborhood_frequencies.max(axis=1)[0]
    # calculate mean purity over all instances
    purity = instancewise_purity.mean().item()
    return purity


def _check_number_of_points(labels_true: np.ndarray, labels_pred: np.ndarray) -> bool:
    """
    Check if the length of the ground truth labels and the prediction labels match.
    If they do not match throw an exception.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    boolean : bool
        True if execution was successful
    """
    if labels_pred.shape[0] != labels_true.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: "
            + str(labels_pred.shape[0])
            + "\nNumber of ground truth objects: "
            + str(labels_true.shape[0])
        )
    return True


def unsupervised_clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Evaluate the quality of predicted labels by comparing it to the ground truth labels using the
    clustering accuracy.
    Returns a value between 1.0 (perfect match) and 0.0 (arbitrary result).
    Since the id of a cluster is not fixed in a clustering setting, the clustering accuracy evaluates each possible
    combination with the ground truth labels.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    acc : float
        The accuracy between the two input label sets.

    References
    -------
    Yang, Yi, et al. "Image clustering using local discriminant models and global integration."
    IEEE Transactions on Image Processing 19.10 (2010): 2761-2773.
    """
    _check_number_of_points(labels_true, labels_pred)
    max_label = int(max(labels_pred.max(), labels_true.max()) + 1)
    match_matrix = np.zeros((max_label, max_label), dtype=np.int64)
    for i in range(labels_true.shape[0]):
        match_matrix[int(labels_true[i]), int(labels_pred[i])] -= 1
    indices = linear_sum_assignment(match_matrix)
    acc = -np.sum(match_matrix[indices]) / labels_pred.size
    return acc


def _get_subsampled_tensors(X: np.array, y: np.array, subsample_size: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    if X.shape[0] > subsample_size:
        # NOTE: Numpy random seed is already set via "set_torch_seed" of train.py. No need to set it again here.
        rnd_indices = np.random.randint(0, X.shape[0], size=subsample_size)
        _subsample_X = X[rnd_indices]
        _subsample_y = y[rnd_indices]
    else:
        _subsample_X = X
        _subsample_y = y
    _subsample_X = torch.from_numpy(_subsample_X)
    _subsample_y = torch.from_numpy(_subsample_y)
    return _subsample_X, _subsample_y


def evaluate_deep_clustering(
    cluster_centers: np.ndarray,
    model: _AbstractAutoencoder,
    dataloader: torch.utils.data.dataloader.DataLoader,
    labels: np.ndarray,
    old_labels: np.ndarray | None,
    loss_fn: torch.nn.Module,
    device: torch.device,
    metrics_dict: dict | None,
    return_labels: bool,
    track_silhouette: bool,
    track_purity: bool,
    track_voronoi: bool,
    track_uncertainty_plot: bool,
) -> tuple[dict, np.ndarray, np.ndarray] | tuple[dict, np.ndarray]:
    """Calculates Accuracy (ACC), Normalized Mutual Information (NMI), Adjusted Rand Index (ARI),
    Autoencoder Reconstruction Error (REC), Local Purity, and cluster label change (0, if old_labels=None) from a centroid based clustering and given model."""
    embeddings, _, rec_loss = encode_decode_batchwise(dataloader, model, device, loss_fn=loss_fn)
    predicted_labels = embedded_kmeans_prediction(dataloader, cluster_centers, model, device=device, embedded_data=embeddings)
    # Considered Metrics
    metrics = {
        "ACC": unsupervised_clustering_accuracy,
        "NMI": normalized_mutual_info_score,
        "ARI": adjusted_rand_score,
        "Cluster_Change_ACC": unsupervised_clustering_accuracy,
        "Cluster_Change_NMI": normalized_mutual_info_score,
        "Cluster_Change_ARI": adjusted_rand_score,
    }

    if track_silhouette:
        metrics |= {
            "SIL": None,
            "Inter_CD": None,
            "Intra_CD": None,
            "Max_Inter_CD": None,
            "Max_Intra_CD": None,
        }
    if track_purity:
        # need a random subsample for calculation of local_purity for large data sets to save gpu memory
        subsample_pt, subsample_labels_pt = _get_subsampled_tensors(embeddings, labels, subsample_size=5000)
        min_cluster_size = np.unique(subsample_labels_pt, return_counts=True)[1].min()
        nn_indices = compute_nearest_neighbor_indices(subsample_pt, nn=None, device=device)
        # torch does not free up the memory of compute_nearest_neighbor_indices, so we do it explicitly
        torch.cuda.empty_cache()
        metrics |= {
            "LocalPurity1": lambda X, y: local_purity_torch(X, y, nn=1, precomputed_nn_indices=nn_indices),
            "LocalPurity10": lambda X, y: local_purity_torch(X, y, nn=10, precomputed_nn_indices=nn_indices),
            "LocalPurity100": lambda X, y: local_purity_torch(X, y, nn=100, precomputed_nn_indices=nn_indices),
            "LocalPurity_MinClusterSize": lambda X, y: local_purity_torch(
                X, y, nn=min_cluster_size, precomputed_nn_indices=nn_indices
            ),
        }

    if track_voronoi:
        voronoi_plot = pca_voronoi_plot(embeddings, predicted_labels)
        voronoi_plot_image = wandb_image(voronoi_plot, caption="Voronoi Plot")
        metrics |= {"Voronoi Plot": voronoi_plot_image}

    uncertainty_score_values = uncertainty_score(embeddings, cluster_centers)
    metrics |= {
        "Uncertainty_Score_Mean": uncertainty_score_values,
        "Uncertainty_Score_Min": uncertainty_score_values,
        "Uncertainty_Score_Max": uncertainty_score_values,
        "Uncertainty_Score_Std": uncertainty_score_values,
    }

    if track_uncertainty_plot:
        plt.hist(uncertainty_score_values, bins=25, range=(0, 1))
        plt.title("Uncertainty Score Distribution Plot")
        plt.xlabel("Uncertainty Score")
        plt.ylabel("Frequency")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        uncertainty_score_plot_image = Image.open(buffer)
        metrics |= {
            "Uncertainty Score Distribution Plot": wandb_image(
                uncertainty_score_plot_image, caption="Uncertainty Score Distribution Plot"
            )
        }
        plt.close()

    # Initialize metrics dict
    if metrics_dict is None:
        metrics_dict = {k: [] for k in metrics}
    # In the initial logging, we have no old_labels yet and thus, old_labels=None, can be passed
    if old_labels is None:
        old_labels = predicted_labels

    # calculate metrics
    for metric_i in metrics.keys():
        if metric_i == "REC":
            metrics_dict[metric_i].append(rec_loss)
        elif "Cluster_Change" in metric_i:
            metrics_dict[metric_i].append(1 - metrics[metric_i](predicted_labels, old_labels))
        elif "LocalPurity" in metric_i:
            metrics_dict[metric_i].append(metrics[metric_i](None, subsample_labels_pt))
        elif "SIL" in metric_i:
            # calculate all metrics at once to save compute
            sil, inter_dist, intra_dist, max_inter_dist, max_intra_dist = silhouette_score(embeddings, labels, metric="cosine")
            metrics_dict["SIL"].append(sil)
            metrics_dict["Inter_CD"].append(inter_dist)
            metrics_dict["Intra_CD"].append(intra_dist)
            metrics_dict["Max_Inter_CD"].append(max_inter_dist)
            metrics_dict["Max_Intra_CD"].append(max_intra_dist)
        elif "Inter_CD" in metric_i or "Intra_CD" in metric_i:
            # already computed in SIL above
            pass
        elif "Voronoi Plot" in metric_i:
            metrics_dict[metric_i].append(metrics[metric_i])
        elif "Uncertainty_Score_Mean" in metric_i:
            metrics_dict["Uncertainty_Score_Mean"].append(np.mean(metrics[metric_i]))
        elif "Uncertainty_Score_Min" in metric_i:
            metrics_dict["Uncertainty_Score_Min"].append(np.min(metrics[metric_i]))
        elif "Uncertainty_Score_Max" in metric_i:
            metrics_dict["Uncertainty_Score_Max"].append(np.max(metrics[metric_i]))
        elif "Uncertainty_Score_Std" in metric_i:
            metrics_dict["Uncertainty_Score_Std"].append(np.std(metrics[metric_i]))
        elif "Uncertainty Score Distribution Plot" in metric_i:
            metrics_dict[metric_i].append(metrics[metric_i])
        else:
            metrics_dict[metric_i].append(metrics[metric_i](labels, predicted_labels))
    if return_labels:
        return metrics_dict, embeddings, predicted_labels
    else:
        return metrics_dict, embeddings
