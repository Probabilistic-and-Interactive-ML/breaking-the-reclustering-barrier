# Allows returning the class type from a method of that class
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import trange

from src.deep._clustering_utils import (
    calculate_supervised_centers,
    squared_euclidean_distance,
)
from src.deep._model_utils import save_model_checkpoint
from src.deep._optimizer_utils import init_lr_scheduler
from src.deep.autoencoders._utils import split_views
from src.deep.evaluation import evaluate_deep_clustering
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ._clustering_utils import apply_kmeans, get_nclusters
from .brb_reclustering import apply_brb


def _int_to_one_hot(int_tensor: torch.Tensor, n_integers: int) -> torch.Tensor:
    """
    Convert a tensor containing integers (e.g. labels) to an one hot encoding.
    Here, each integer gets its own features in the resulting tensor, where only the values 0 or 1 are accepted.
    E.g. [0,0,1,2,1] gets
    [[1,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]]

    Parameters
    ----------
    int_tensor : torch.Tensor
        The original tensor containing integers
    n_integers : int
        The number of different integers within int_tensor

    Returns
    -------
    onehot : torch.Tensor
        The final one hot encoding tensor
    """
    onehot = torch.zeros([int_tensor.shape[0], n_integers], dtype=torch.float, device=int_tensor.device)
    onehot.scatter_(1, int_tensor.unsqueeze(1).long(), 1)
    return onehot


# DCN is reinitializing clusters if they are lost
# See Row 1071 at https://github.com/boyangumn/DCN-New/blob/master/multi_layer_km.py
def get_lost_cluster_indices(assignments, n_classes):
    set_classes = set([i for i in range(n_classes)])
    set_assigned = set(assignments.tolist())
    diff = set_assigned.symmetric_difference(set_classes)
    return list(diff)


# Common reinit strategies
def random_reinit_centroids(lost_cluster_indices, embedded, noise_magnitude=1e-6):
    rand_indices = np.random.randint(low=0, high=embedded.shape[0], size=len(lost_cluster_indices))
    random_perturbation = torch.empty_like(embedded[rand_indices]).normal_(
        mean=embedded.mean().item(), std=embedded.std().item()
    )
    return embedded[rand_indices] + noise_magnitude * random_perturbation


def _compute_centroids_mb(
    centers: torch.Tensor,
    embedded: torch.Tensor,
    count: torch.Tensor,
    labels: torch.Tensor,
    center_lr: float = 0.5,
    weights: torch.Tensor = None,
    reinit_lost_clusters: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the centers and amount of object ever assigned to a center.

    New center is calculated by (see Eq. 8 in the paper):
    center - eta (center - embedded[i])
    => center - eta * center + eta * embedded[i]
    => (1 - eta) center + eta * embedded[i]

    Parameters
    ----------
    centers : torch.Tensor
        The current cluster centers
    embedded : torch.Tensor
        The embedded samples
    count : torch.Tensor
        The total amount of objects that ever got assigned to a cluster. Affects the learning rate of the center update
    labels : torch.Tensor
        The current hard labels
    center_lr : float
        determines how fast new centers are updated based on new mini-batch and the counts (default: 0.5)
    weights : torch.Tensor
        weights dimensions for euclidean distance calculation (default: None)
    reinit_lost_clusters : bool
        If True a lost cluster in a mini-batch will trigger the reinitialization procedure.
        Can lead to instabilities for small mini-batch sizes (default: False)

    Returns
    -------
    centers, count : (torch.Tensor, torch.Tensor)
        The updated centers and the updated counts
    """
    # Minibatch variant of DCN update
    n_clusters = centers.shape[0]
    lost_cluster_indices = get_lost_cluster_indices(assignments=labels, n_classes=len(centers))
    assignment_matrix = _int_to_one_hot(labels, n_clusters)
    copy_count = count.clone().unsqueeze(1).float()
    if reinit_lost_clusters:
        # Reset if cluster is lost and count for a clusters is less than the expected value over a minibatch (better-reinit tag)
        if len(lost_cluster_indices) > 0 and count[lost_cluster_indices] < (embedded.shape[0] // n_clusters + 1):
            centers[lost_cluster_indices] = random_reinit_centroids(lost_cluster_indices, embedded)
            # # Reset count (selected-count-reset tag)
            count[lost_cluster_indices] = (torch.ones(n_clusters) * 100).int().to(embedded.device)[lost_cluster_indices]
            copy_count = count.clone().unsqueeze(1).float()

            # # recalculate assignments
            dist = squared_euclidean_distance(embedded, centers, weights=weights)
            _new_labels = dist.min(dim=1)[1]
            assignment_matrix = _int_to_one_hot(_new_labels, n_clusters)
    batch_cluster_sums = (embedded.unsqueeze(1).detach() * assignment_matrix.unsqueeze(2)).sum(0)
    mask_sum = assignment_matrix.sum(0).unsqueeze(1)
    nonzero_mask = mask_sum.squeeze(1) != 0

    copy_count[nonzero_mask] = (1 - center_lr) * copy_count[nonzero_mask]
    copy_count[nonzero_mask] += center_lr * mask_sum[nonzero_mask]
    per_center_lr = 1.0 / (copy_count[nonzero_mask] + 1)
    batch_centers = batch_cluster_sums[nonzero_mask] / mask_sum[nonzero_mask]
    centers[nonzero_mask] = (1.0 - per_center_lr) * centers[nonzero_mask] + per_center_lr * batch_centers

    # squeeze and convert copy_count back to int
    copy_count = copy_count.squeeze(1).int()
    return centers, copy_count


def _compute_centroids_per_instance(
    centers: torch.Tensor, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the centers and amount of object ever assigned to a center.

    New center is calculated by (see Eq. 8 in the paper):
    center - eta (center - embedded[i])
    => center - eta * center + eta * embedded[i]
    => (1 - eta) center + eta * embedded[i]

    Parameters
    ----------
    centers : torch.Tensor
        The current cluster centers
    embedded : torch.Tensor
        The embedded samples
    count : torch.Tensor
        The total amount of objects that ever got assigned to a cluster. Affects the learning rate of the center update
    labels : torch.Tensor
        The current hard labels

    Returns
    -------
    centers, count : (torch.Tensor, torch.Tensor)
        The updated centers and the updated counts
    """
    for i in range(embedded.shape[0]):
        c = labels[i].item()
        count[c] += 1
        eta = 1.0 / count[c].item()
        centers[c] = (1 - eta) * centers[c] + eta * embedded[i]
    return centers, count


class _BRB_DCN_Module(torch.nn.Module):
    """
    The _BRB_DCN_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_np_centers : np.ndarray
        The initial numpy centers
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)


    Attributes
    ----------
    init_np_centers : torch.Tensor
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    use_mb_center_update : bool
        If True then centers are updated in mini-batch fashion. Otherwise update is done per instance (slower on GPU)
    n_clusters: int
        Number of clusters
    """

    def __init__(
        self,
        init_np_centers: np.ndarray,
        ground_truth_labels: list[int],
        augmentation_invariance: bool = False,
        use_mb_center_update: bool = True,
    ):
        super().__init__()
        self.augmentation_invariance = augmentation_invariance
        self.centers = torch.tensor(init_np_centers)
        self.use_mb_center_update = use_mb_center_update
        self.n_clusters = init_np_centers.shape[0]

        # initialize metrics for tracking the training
        self.metrics = None
        self.ground_truth_labels = np.array(ground_truth_labels)

    def dcn_loss(
        self, embedded: torch.Tensor, assignment_matrix: torch.Tensor = None, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculate the DCN loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        assignment_matrix : torch.Tensor
            cluster assignments per sample as a one-hot-matrix to compute the loss.
            If None then loss will be computed based on the closest centroids for each data sample (default: None)
        weights : torch.Tensor
            feature weights for the squared euclidean distance (default: None)

        Returns
        -------
        loss: torch.Tensor
            the final DCN loss
        """
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        if assignment_matrix is None:
            loss = (dist.min(dim=1)[0]).mean()
        else:
            loss = (dist * assignment_matrix).mean()
        return loss

    def predict_hard(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the minimum squared euclidean distance to the cluster centers to get the labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance (default: None)

        Returns
        -------
        labels : torch.Tensor
            the final labels
        """
        dist = squared_euclidean_distance(embedded, self.centers, weights=weights)
        labels = dist.min(dim=1)[1]
        return labels

    def update_centroids(
        self, embedded: torch.Tensor, count: torch.Tensor, labels: torch.Tensor, reinit_lost_clusters: bool = False
    ) -> torch.Tensor:
        """
        Update the cluster centers of the _DCN_Module.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        count : torch.Tensor
            The total amount of objects that ever got assigned to a cluster. Affects the learning rate of the center update
        labels : torch.Tensor
            The current hard labels
        reinit_lost_clusters : bool
            If True a lost cluster in a mini-batch will trigger the reinitialization procedure.
            Can lead to instabilities for small mini-batch sizes (default: False)

        Returns
        -------
        count : torch.Tensor
            The new amount of objects that ever got assigned to a cluster
        """
        if self.use_mb_center_update:
            self.centers, count = _compute_centroids_mb(self.centers, embedded, count, labels, reinit_lost_clusters)
        else:
            self.centers, count = _compute_centroids_per_instance(self.centers, embedded, count, labels)
        return count

    def to_device(self, device: torch.device) -> _BRB_DCN_Module:
        """
        Move the _DCN_Module and the cluster centers to the specified device (cpu or cuda).

        Parameters
        ----------
        device : torch.device
            device to be trained on

        Returns
        -------
        self : _DCN_Module
            this instance of the _DCN_Module
        """
        self.centers = self.centers.to(device)
        self.to(device)
        return self

    def cluster_loss(self, indices, embedded, embedded_aug=None):
        """
        Calculate the DCN cluster loss of given embedded samples.

        Parameters
        ----------
        indices: torch.Tensor
            mini-batch indices
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor | None
            the embedded samples of augmented data (only needed if self.augmentation_invariance=True)

        Returns
        -------
        cluster_loss : torch.Tensor
            the DCN cluster loss
        """
        assignments = self.predict_hard(embedded)
        assignment_matrix = _int_to_one_hot(assignments, self.centers.shape[0])

        cluster_loss = self.dcn_loss(embedded, assignment_matrix)

        if self.augmentation_invariance:
            # assign augmented samples to the same cluster as original samples
            cluster_loss_aug = self.dcn_loss(embedded_aug, assignment_matrix)
            cluster_loss = (cluster_loss + cluster_loss_aug) / 2

        return cluster_loss

    def set_centers(self, centers: torch.Tensor):
        device = self.centers.device
        assert self.centers.shape == centers.shape
        self.centers = torch.tensor(centers).to(device)

    def update_centers_with_labels(
        self,
        labels: np.array,
        autoencoder: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        embeddings: np.array,
    ):
        if get_nclusters(labels) < self.n_clusters:
            # cluster(s) have been lost, need to recluster
            _, _new_labels = apply_kmeans(embeddings, self.n_clusters)
            _new_centers = calculate_supervised_centers(_new_labels, autoencoder, dataloader, device)
        else:
            _new_centers = calculate_supervised_centers(labels, autoencoder, dataloader, device)
        self.set_centers(_new_centers)

    def _loss(
        self,
        batch: list,
        autoencoder: torch.nn.Module,
        cluster_loss_weight: float,
        reconstruction_loss_weight: float,
        loss_fn: torch.nn.modules.loss._Loss,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate the complete DCN + Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        autoencoder : torch.nn.Module
            the autoencoder
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss
        reconstruction_loss_weight : float
            weight of the reconstruction loss
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        clust_loss_dict : dict[str, torch.Tensor]
            A dictionary holding loss and training metrics for DCN
        """

        # initialize values
        # init with None to avoid errors if augmentation is not used
        embedded_aug = None
        clust_loss_dict = {"loss": torch.tensor(0.0, device=device)}
        indices = batch[0]
        # compute reconstruction loss
        if self.augmentation_invariance:
            # Both batches will be merged in the autoencoder loss and need to be split again (code below)
            loss_dict, embedded, _ = autoencoder.loss(batch, loss_fn, device)
            # Convention is that the augmented sample is at the first position and the original one at the second position
            embedded_aug, embedded = split_views(embedded)
            # Add loss
            clust_loss_dict["loss"] += loss_dict["loss"] * reconstruction_loss_weight
        else:
            loss_dict, embedded, _ = autoencoder.loss(batch, loss_fn, device)
            clust_loss_dict["loss"] += loss_dict["loss"] * reconstruction_loss_weight

        # Update the loss dict with the AE loss and metrics
        clust_loss_dict |= {f"AE_{k}": v for k, v in loss_dict.items()}

        # Compute Cluster loss
        cluster_loss = self.cluster_loss(indices, embedded, embedded_aug)

        clust_loss_dict["loss"] += cluster_loss * cluster_loss_weight

        clust_loss_dict["cluster_loss"] = cluster_loss

        return clust_loss_dict

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        # newly added for BRB DCN
        trainloader_wA: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        n_epochs: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_args: dict,
        loss_fn: torch.nn.modules.loss._Loss,
        cluster_loss_weight: float,
        reconstruction_loss_weight: float,
        reset_args: dict[str, int | float | bool],
        checkpoint_path: str = None,
        checkpointing_frequency: int = 10,
        wandb_run: Run | RunDisabled | None = None,
        log_interval: int = 1,
        track_silhouette: bool = False,
        track_purity: bool = False,
        track_voronoi: bool = False,
        track_uncertainty_plot: bool = False,
    ) -> _BRB_DCN_Module:
        """
        Trains the _DCN_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            the autoencoder
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        evalloader : torch.utils.data.DataLoader
            dataloader to be used for evaluation (assumes shuffle = False)
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        cluster_loss_weight : float
            weight of the clustering loss
        reconstruction_loss_weight : float
            weight of the reconstruction loss
        checkpoint_path : str
            path to where model checkpoints should be saved (default: None)
        checkpointing_frequency : int
            frequency with which model checkpoints should be saved (default: 10)
        reset_args: dict[str, int | float | bool]
            Dictionary containing the reset parameters
        wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
            wandb run object for logging clustering training progress (default: None)
        log_interval: int
            Interval at which clustering metrics should be logged (default: 1)
        track_silhouette: bool
            If True, Inter-CD, Intra-DC and Silhouette score will be calculated and logged during training (default: False)
        track_purity: bool
            If True, local purity will be calculated and logged during training (default: False)
        track_voronoi: bool
            If True, the voronoi diagram will be calculated and logged during training (default: False)
        track_uncertainty_plot: bool
            If True, the uncertainty plot will be calculated and logged during training (default: False)

        Returns
        -------
        self : _DCN_Module
            this instance of the _DCN_Module
        """

        # log initial performance
        self.metrics, _, _predicted_labels = evaluate_deep_clustering(
            cluster_centers=self.centers.detach().cpu().numpy(),
            model=autoencoder,
            dataloader=testloader,
            labels=self.ground_truth_labels,
            old_labels=None,
            loss_fn=loss_fn,
            device=device,
            metrics_dict=self.metrics,
            return_labels=True,
            track_silhouette=track_silhouette,
            track_purity=track_purity,
            track_voronoi=track_voronoi,
            track_uncertainty_plot=track_uncertainty_plot,
        )

        # Log initial scores before the DC algorithm trains
        if wandb_run is not None:
            metric_dict = {f"Clustering metrics/{k}": v[-1] for k, v in self.metrics.items()}
            metric_dict["Clustering epoch"] = 0
            wandb_run.log(metric_dict)

        save_model_checkpoint(checkpoint_path, checkpointing_frequency, -1, n_epochs, autoencoder, self.centers, self.metrics)

        # Init for count from original DCN code (not reported in Paper)
        # This means centroid learning rate at the beginning is scaled by a hundred
        count = (torch.ones(self.centers.shape[0], dtype=torch.int32) * 100).to(device)

        # Init lr_scheduler
        lr_scheduler = init_lr_scheduler(optimizer, lr_scheduler_args)

        # DCN training loop
        pbar = trange(n_epochs, desc="DCN Training")
        for epoch_i in pbar:
            if epoch_i != 0 and (epoch_i % reset_args.reset_interval) == 0:
                lr_scheduler_args.scheduler_warmup_epochs += epoch_i
                # NOTE: For DCN, centroid momentum is never reset
                autoencoder, optimizer, _ = apply_brb(
                    cluster_algorithm=self,
                    autoencoder=autoencoder,
                    optimizer=optimizer,
                    reset_args=reset_args,
                    reset_momentum=False,
                    dataloader=trainloader_wA,
                    device=device,
                )

            epoch_log_dict = defaultdict(float)

            if reset_args.batchnorm_eval_mode:
                autoencoder.eval()
            else:
                # Enable autoencoder train mode such that batchnorm is properly updated
                autoencoder.train()
            # Update Network
            for batch in trainloader:
                loss_dict = self._loss(
                    batch=batch,
                    autoencoder=autoencoder,
                    cluster_loss_weight=cluster_loss_weight,
                    reconstruction_loss_weight=reconstruction_loss_weight,
                    loss_fn=loss_fn,
                    device=device,
                )
                # Backward pass - update weights
                optimizer.zero_grad()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    loss_dict["loss"].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 500, norm_type=2.0)

                # Logging
                l0_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1e10, norm_type=0)
                l1_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1e10, norm_type=1.0)
                l2_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1e10, norm_type=2.0)
                optimizer.step()

                if wandb_run is not None:
                    for k, v in loss_dict.items():
                        epoch_log_dict[f"Clustering train/{k}"] += v.item()

                    epoch_log_dict["Clustering train/l0_grad_norm"] += l0_grad_norm
                    epoch_log_dict["Clustering train/l1_grad_norm"] += l1_grad_norm
                    epoch_log_dict["Clustering train/l2_grad_norm"] += l2_grad_norm

            autoencoder.eval()
            # Update Assignments and Centroids
            with torch.no_grad():
                for batch in trainloader:
                    if self.augmentation_invariance:
                        # Convention is that the augmented sample is at the first position and the original one at the second position
                        # We only use the original sample for updating the centroids and assignments
                        batch_data = batch[2].to(device)
                    else:
                        batch_data = batch[1].to(device)
                    embedded = autoencoder.encode(batch_data)
                    # update assignments
                    labels = self.predict_hard(embedded)
                    # update centroids
                    count = self.update_centroids(embedded, count, labels)

            # Evaluation:
            if epoch_i % log_interval == 0 or epoch_i == n_epochs - 1:
                self.metrics, _, _predicted_labels = evaluate_deep_clustering(
                    cluster_centers=self.centers.detach().cpu().numpy(),
                    model=autoencoder,
                    dataloader=testloader,
                    labels=self.ground_truth_labels,
                    old_labels=_predicted_labels,
                    loss_fn=loss_fn,
                    device=device,
                    metrics_dict=self.metrics,
                    return_labels=True,
                    track_silhouette=track_silhouette,
                    track_purity=track_purity,
                    track_voronoi=track_voronoi,
                    track_uncertainty_plot=track_uncertainty_plot,
                )

                # Add the clustering metrics
                if wandb_run is not None:
                    # Compute epoch averages
                    for k, v in epoch_log_dict.items():
                        epoch_log_dict[k] = v / len(trainloader)
                    epoch_log_dict |= {f"Clustering metrics/{k}": v[-1] for k, v in self.metrics.items()}
                    epoch_log_dict["Clustering epoch"] = epoch_i + 1
                    if lr_scheduler is not None:
                        epoch_log_dict["Clustering train/lr"] = optimizer.param_groups[0]["lr"]
                    wandb_run.log(epoch_log_dict)

                pbar.set_postfix_str(f"ACC: {(100*self.metrics['ACC'][-1]):.2f} | ARI: {(100*self.metrics['ARI'][-1]):.2f}")

                save_model_checkpoint(
                    checkpoint_path, checkpointing_frequency, epoch_i, n_epochs, autoencoder, self.centers, self.metrics
                )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch_i)

        return self
