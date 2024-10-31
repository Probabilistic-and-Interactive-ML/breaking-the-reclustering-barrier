from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import trange

from config.optimizer_config import DCOptimizerArgs
from src.deep._clustering_utils import squared_euclidean_distance
from src.deep._model_utils import save_model_checkpoint
from src.deep._optimizer_utils import init_lr_scheduler
from src.deep.autoencoders._utils import split_views
from src.deep.evaluation import evaluate_deep_clustering
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from .brb_reclustering import apply_brb


def _dec_predict(centers: torch.Tensor, embedded: torch.Tensor, alpha: float, weights) -> torch.Tensor:
    """
    Predict soft cluster labels given embedded samples.

    Parameters
    ----------
    centers : torch.Tensor
        the cluster centers
    embedded : torch.Tensor
        the embedded samples
    alpha : float
        the alpha value
    weights : torch.Tensor
        feature weights for the squared euclidean distance (default: None)


    Returns
    -------
    prob : torch.Tensor
        The predicted soft labels
    """
    squared_diffs = squared_euclidean_distance(embedded, centers, weights)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob


def _dec_compression_value(pred_labels: torch.Tensor) -> torch.Tensor:
    """
    Get the DEC compression values.

    Parameters
    ----------
    pred_labels : torch.Tensor
        the predictions of the embedded samples.

    Returns
    -------
    p : torch.Tensor
        The compression values
    """
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p


def _dec_compression_loss_fn(pred_labels: torch.Tensor, target_p: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the loss of DEC by computing the DEC compression value.

    Parameters
    ----------
    pred_labels : torch.Tensor
        the predictions of the embedded samples.
    target_p : torch.Tensor
        dec_compression_value used as pseudo target labels

    Returns
    -------
    loss : torch.Tensor
        The final loss
    """
    if target_p is None:
        target_p = _dec_compression_value(pred_labels).detach().data
    loss = -1.0 * torch.mean(torch.sum(target_p * torch.log(pred_labels + 1e-8), dim=1))
    return loss


class _BRB_DEC_Module(torch.nn.Module):
    """
    The _BRB_DEC_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_np_centers : np.ndarray
        The initial cluster centers
    alpha : double
        alpha value for the prediction method
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    alpha : float
        the alpha value
    centers : torch.Tensor:
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    ground_truth_labels: list,
        ground truth labels used for evaluation
    """

    def __init__(
        self, init_np_centers: np.ndarray, ground_truth_labels: list[int], alpha: float, augmentation_invariance: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.augmentation_invariance = augmentation_invariance
        # Centers are learnable parameters
        self.centers = torch.nn.Parameter(torch.tensor(init_np_centers), requires_grad=True)
        self.n_clusters = init_np_centers.shape[0]

        # initialize metrics for tracking the training
        self.metrics = None
        self.ground_truth_labels = np.array(ground_truth_labels)

    def predict(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Soft prediction of given embedded samples. Returns the corresponding soft labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        pred = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        return pred

    def predict_hard(self, embedded: torch.Tensor, weights=None) -> torch.Tensor:
        """
        Hard prediction of the given embedded samples. Returns the corresponding hard labels.
        Uses the soft prediction method and then applies argmax.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred_hard : torch.Tensor
            The predicted hard labels
        """
        pred_hard = self.predict(embedded, weights=weights).argmax(1)
        return pred_hard

    def set_centers(self, centers: torch.Tensor):
        """Overwrites learnable centers by new centers"""
        assert self.centers.shape == centers.shape
        device = self.centers.device
        with torch.no_grad():
            self.centers.data = torch.tensor(centers)
        # load centers to device
        self.to(device)

    def dec_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        prediction = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        loss = _dec_compression_loss_fn(prediction)
        return loss

    def dec_augmentation_invariance_loss(
        self, embedded: torch.Tensor, embedded_aug: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples with augmentation invariance.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        weights : torch.Tensor
            feature weights for the squared euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        prediction = _dec_predict(self.centers, embedded, self.alpha, weights=weights)
        # Predict pseudo cluster labels with clean samples
        clean_target_p = _dec_compression_value(prediction).detach().data
        # Calculate loss from clean prediction and clean targets
        clean_loss = _dec_compression_loss_fn(prediction, clean_target_p)

        # Predict pseudo cluster labels with augmented samples
        aug_prediction = _dec_predict(self.centers, embedded_aug, self.alpha, weights=weights)
        # Calculate loss from augmented prediction and reused clean targets to enforce that the
        # cluster assignment is invariant against augmentations
        aug_loss = _dec_compression_loss_fn(aug_prediction, clean_target_p)

        # average losses
        loss = (clean_loss + aug_loss) / 2
        return loss

    def _loss(
        self,
        batch: list,
        autoencoder: torch.nn.Module,
        cluster_loss_weight: float,
        data_reg_loss_weight: float,
        use_reconstruction_loss: bool,
        loss_fn: torch.nn.modules.loss._Loss,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """
        Calculate the complete DEC + optional Autoencoder loss.

        Parameters
        ----------
        batch : list
            the minibatch
        autoencoder : torch.nn.Module
            the autoencoder
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss
        data_reg_loss_weight : float
            weight of the regularization loss (reconstruction or contrastive loss) loss compared clustering loss
        use_reconstruction_loss : bool
            defines whether the reconstruction loss will be used during clustering training
        loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        clust_loss_dict : dict[str, torch.Tensor]
            A dictionary holding loss and training metrics for DEC
        """

        # initialize values
        clust_loss_dict = {"loss": torch.tensor(0.0, device=device)}
        # Reconstruction loss is not included in DEC
        if use_reconstruction_loss:
            if self.augmentation_invariance:
                # Both batches will be merged in the autoencoder loss and need to be split again (code below)
                loss_dict, embedded, _ = autoencoder.loss(batch, loss_fn, device)
                # Convention is that the augmented sample is at the first position and the original one at the second position
                embedded_aug, embedded = split_views(embedded)
                clust_loss_dict["loss"] += data_reg_loss_weight * loss_dict["loss"]
            else:
                batch_data = batch[1].to(device)
                loss_dict, embedded, _ = autoencoder.loss([batch[0], batch_data], loss_fn, device)
                clust_loss_dict["loss"] += data_reg_loss_weight * loss_dict["loss"]

            # Update the loss dict with the AE loss and metrics
            clust_loss_dict |= {f"AE_{k}": v for k, v in loss_dict.items()}
        else:
            # Generate some metrics to see what happens with the AE during clustering
            with torch.inference_mode():
                loss_dict, _, _ = autoencoder.loss(batch, loss_fn, device)
            clust_loss_dict |= {f"AE_{k}": v for k, v in loss_dict.items()}

            if self.augmentation_invariance:
                aug_data = batch[1].to(device)
                embedded_aug = autoencoder.encode(aug_data)
                batch_data = batch[2].to(device)
                embedded = autoencoder.encode(batch_data)
            else:
                batch_data = batch[1].to(device)
                embedded = autoencoder.encode(batch_data)

        # CLuster loss
        if self.augmentation_invariance:
            cluster_loss = self.dec_augmentation_invariance_loss(embedded, embedded_aug)
        else:
            cluster_loss = self.dec_loss(embedded)
        clust_loss_dict["loss"] += cluster_loss * cluster_loss_weight

        clust_loss_dict["cluster_loss"] = cluster_loss

        return clust_loss_dict

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        trainloader_wA: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        n_epochs: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_args: DCOptimizerArgs,
        loss_fn: torch.nn.modules.loss._Loss,
        use_reconstruction_loss: bool,
        cluster_loss_weight: float,
        data_reg_loss_weight: float,
        reset_args=dict[str, int | float | bool],
        checkpoint_path: str = None,
        checkpointing_frequency: int = 10,
        wandb_run: Run | RunDisabled | None = None,
        log_interval: int = 1,
        track_silhouette: bool = False,
        track_purity: bool = False,
        track_voronoi: bool = False,
        track_uncertainty_plot: bool = False,
    ) -> _BRB_DEC_Module:
        """
        Trains the _DEC_Module in place.

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
        use_reconstruction_loss : bool
            defines whether the reconstruction loss will be used during clustering training
        cluster_loss_weight : float
            weight of the clustering loss compared to the reconstruction loss
        data_reg_loss_weight : float
            weight of the regularization loss (reconstruction or contrastive loss) loss compared clustering loss
        reset_args: dict[str, int | float | bool]
            Parameters for the reset methods
        checkpoint_path : str
            path to where model checkpoints should be saved (default: None)
        checkpointing_frequency : int
            frequency with which model checkpoints should be saved (default: 10)
        wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
            wandb run object for logging clustering training progress (default: None)
        log_interval: int
            Interval at which clustering metrics should be logged (default: 1)
        reset_interval : int
            interval for resets is defined as a percentage of total steps.
        track_silhouette: bool
            If True, Inter-CD, Intra-DC and Silhouette score will be calculated and logged during training (default: False)
        track_purity: bool
            If True, local purity will be calculated and logged during training (default: False)
        track_voronoi: bool
            If True, Voronoi plot will be calculated and logged during training (default: False)
        track_uncertainty_plot: bool
            If True, uncertainty plot will be calculated and logged during training (default: False)

        Returns
        -------
        self : _DEC_Module
            this instance of the _DEC_Module
        """

        # log initial performance
        self.metrics, _, _predicted_labels = evaluate_deep_clustering(
            cluster_centers=self.centers.detach().cpu().numpy(),
            model=autoencoder,
            dataloader=testloader,
            labels=self.ground_truth_labels,
            old_labels=None,
            loss_fn=None,
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
        pbar = trange(n_epochs, desc=f"{'IDEC' if use_reconstruction_loss else 'DEC'} Training")

        # Init lr scheduler
        lr_scheduler = init_lr_scheduler(optimizer, lr_scheduler_args)
        for epoch_i in pbar:
            if epoch_i != 0 and (epoch_i % reset_args.reset_interval) == 0:
                lr_scheduler_args.scheduler_warmup_epochs += epoch_i
                # NOTE: For DEC/IDEC we must reset momentum as the centers are learnable parameters
                autoencoder, optimizer, _ = apply_brb(
                    cluster_algorithm=self,
                    autoencoder=autoencoder,
                    optimizer=optimizer,
                    reset_args=reset_args,
                    reset_momentum=reset_args.reset_momentum,
                    dataloader=trainloader_wA,
                    device=device,
                )

            if reset_args.batchnorm_eval_mode:
                autoencoder.eval()
            else:
                # Enable autoencoder train mode such that batchnorm is properly updated
                autoencoder.train()
            epoch_log_dict = defaultdict(float)
            for batch in trainloader:
                loss_dict = self._loss(
                    batch=batch,
                    autoencoder=autoencoder,
                    cluster_loss_weight=cluster_loss_weight,
                    data_reg_loss_weight=data_reg_loss_weight,
                    use_reconstruction_loss=use_reconstruction_loss,
                    loss_fn=loss_fn,
                    device=device,
                )

                # Backward pass
                optimizer.zero_grad()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    loss_dict["loss"].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 500, norm_type=2.0)

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

                pbar.set_postfix_str(
                    f"ACC: {(100 * self.metrics['ACC'][-1]):.2f} | ARI: {(100 * self.metrics['ARI'][-1]):.2f}"
                )

                save_model_checkpoint(
                    checkpoint_path, checkpointing_frequency, epoch_i, n_epochs, autoencoder, self.centers, self.metrics
                )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch_i + 1)

        return self
