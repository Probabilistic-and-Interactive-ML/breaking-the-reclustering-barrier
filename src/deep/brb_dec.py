from __future__ import annotations

from typing import Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

from config.brb_config import BRBArgs
from config.optimizer_config import DCOptimizerArgs
from src.deep._clustering_utils import (
    encode_batchwise,
    predict_batchwise,
    run_clustering,
)
from src.deep._data_utils import augmentation_invariance_check, get_dataloader
from src.deep._model_utils import get_trained_autoencoder
from src.deep._optimizer_utils import setup_optimizer
from src.deep._torch_utils import detect_device
from src.deep.dec_module import _BRB_DEC_Module
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ._clustering_utils import embedded_kmeans_prediction


class BRB_DEC(BaseEstimator, ClusterMixin):
    """
    The Deep Embedded Clustering (DEC) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DEC loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    checkpoint_path : str
        path to where model checkpoints should be saved (default: None)
    checkpointing_frequency : int
        frequency with which model checkpoints should be saved (default: 10)
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : DCOptimizerArgs
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1.0)
    data_reg_loss_weight : float
        weight of the regularization loss (reconstruction or contrastive loss) loss compared clustering loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    device : torch.device
        device to be trained on (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_cluster_centers : np.array
        initial cluster_centers to be used for DEC (default: None).
    reset_args: dict[str, int | float | bool]
        Various parameters for the reset procedure. See brb_reclustering for details.
    wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
        wandb run object for logging clustering training progress (default: None)
    log_interval: int
        Interval at which clustering metrics should be logged (default: 1)
    num_workers: int
        number of workers used for dataloader (default: 1)
    track_silhouette: bool
        If True, Inter-CD, Intra-DC and Silhouette score will be calculated and logged during training (default: False)
    track_purity: bool
        If True, local purity will be calculated and logged during training (default: False)
    track_voronoi: bool
        If True, Voronoi plots will be calculated and logged during training (default: False)
    track_uncertainty_plot: bool
        If True, uncertainty plot will be calculated and logged during training (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dc_labels_ : np.ndarray
        The final DEC labels
    dc_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder
    metrics : dict
        Dictionary with various performance metrics calculated during training
    wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
        wandb run object for logging clustering training progress (default: None)
    reset_args: dict[str, int | float | bool]

    Examples
    ----------
    >>> from src.datasets import create_subspace_data
    >>> from src.deep import BRB_DEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50))
    >>> dec = BRB_DEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> dec.fit(data)

    References
    ----------
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. 2016.
    """

    def __init__(
        self,
        n_clusters: int,
        checkpoint_path: str = None,
        checkpointing_frequency: int = 10,
        initial_cluster_centers: np.array = None,
        alpha: float = 1.0,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_optimizer_params: DCOptimizerArgs = DCOptimizerArgs(),
        clustering_lr_scheduler_args: dict = {},
        pretrain_epochs: int = 100,
        clustering_epochs: int = 150,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: torch.nn.Module = None,
        embedding_size: int = 10,
        cluster_loss_weight: float = 1.0,
        data_reg_loss_weight: float = 1.0,
        custom_dataloaders: tuple = None,
        device: torch.device = None,
        augmentation_invariance: bool = False,
        reset_args: BRBArgs = BRBArgs(),
        wandb_run: Run | RunDisabled | None = None,
        log_interval: int = 1,
        num_workers: int = 1,
        track_silhouette: bool = False,
        track_purity: bool = False,
        track_voronoi: bool = False,
        track_uncertainty_plot: bool = False,
    ):
        self.n_clusters = n_clusters
        self.checkpoint_path = checkpoint_path
        self.checkpointing_frequency = checkpointing_frequency
        self.metrics = None
        self.initial_cluster_centers = initial_cluster_centers
        self.device = device
        self.alpha = alpha
        self.batch_size = batch_size
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.clustering_lr_scheduler_args = clustering_lr_scheduler_args
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.data_reg_loss_weight = data_reg_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.use_reconstruction_loss = False
        self.num_workers = num_workers

        # BRB parameters
        self.reset_args = reset_args

        # Logging
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.track_silhouette = track_silhouette
        self.track_purity = track_purity
        self.track_voronoi = track_voronoi
        self.track_uncertainty_plot = track_uncertainty_plot

    def fit(self, X: np.ndarray | None, y: np.ndarray) -> Union[BRB_DEC, BRB_IDEC]:
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set. Can be None if custom_dataloaders is specified
        y : np.ndarray
            the labels used for validation

        Returns
        -------
        self : DEC
            this instance of the DEC algorithm
        """
        if self.autoencoder is not None and self.autoencoder.use_contrastive_loss:
            if not self.augmentation_invariance:
                raise ValueError("If contrastive_loss is used, then augmentation_invariance needs to be set to True.")
        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)

        if self.custom_dataloaders is None and X is None:
            raise ValueError("Input data X and custom_dataloaders cannot be both None")

        if self.device is None:
            self.device = detect_device()
        if self.custom_dataloaders is None:
            trainloader = get_dataloader(X, self.batch_size, True, False, dl_kwargs={"num_workers": self.num_workers})
            testloader = get_dataloader(X, self.batch_size, False, False)
        else:
            trainloader, trainloader_wA, testloader = self.custom_dataloaders

        autoencoder = get_trained_autoencoder(
            trainloader,
            self.pretrain_optimizer_params,
            self.pretrain_epochs,
            self.device,
            self.optimizer_class,
            self.loss_fn,
            self.embedding_size,
            self.autoencoder,
        )

        # Execute initial clustering in embedded space
        if self.initial_cluster_centers is None:
            init_centers = run_clustering(
                method=self.reset_args.reclustering_method,
                n_clusters=self.n_clusters,
                autoencoder=autoencoder,
                dataloader=trainloader_wA,
                device=self.device,
                embedded=None,
                subsample_size=None,
            )
        else:
            assert self.initial_cluster_centers.shape[0] == self.n_clusters
            init_centers = self.initial_cluster_centers
        # Setup DEC Module
        dec_module = _BRB_DEC_Module(
            init_np_centers=init_centers,
            alpha=self.alpha,
            ground_truth_labels=y,
            augmentation_invariance=self.augmentation_invariance,
        ).to(self.device)

        # Use DEC optimizer parameters
        optimizer = setup_optimizer(
            model=autoencoder,
            dc_module=dec_module,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.clustering_optimizer_params,
            freeze_convlayers=self.clustering_optimizer_params.freeze_convlayers,
        )

        # DEC Training loop
        dec_module.fit(
            autoencoder=autoencoder,
            trainloader=trainloader,
            trainloader_wA=trainloader_wA,
            testloader=testloader,
            n_epochs=self.clustering_epochs,
            device=self.device,
            optimizer=optimizer,
            lr_scheduler_args=self.clustering_optimizer_params,
            loss_fn=self.loss_fn,
            use_reconstruction_loss=self.use_reconstruction_loss,
            cluster_loss_weight=self.cluster_loss_weight,
            data_reg_loss_weight=self.data_reg_loss_weight,
            checkpoint_path=self.checkpoint_path,
            checkpointing_frequency=self.checkpointing_frequency,
            reset_args=self.reset_args,
            wandb_run=self.wandb_run,
            log_interval=self.log_interval,
            track_silhouette=self.track_silhouette,
            track_purity=self.track_purity,
            track_voronoi=self.track_voronoi,
            track_uncertainty_plot=self.track_uncertainty_plot,
        )
        # Get labels
        self.dc_labels_ = predict_batchwise(trainloader_wA, autoencoder, dec_module, self.device)
        self.dc_cluster_centers_ = dec_module.centers.detach().cpu().numpy()
        self.metrics = dec_module.metrics

        # Do reclustering with Kmeans
        embedded_data = encode_batchwise(trainloader_wA, autoencoder, self.device)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(embedded_data)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_

        self.autoencoder = autoencoder

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        dataloader = get_dataloader(X, self.batch_size, False, False, dl_kwargs={"num_workers": self.num_workers})
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, self.autoencoder, device=self.device)
        return predicted_labels


class BRB_IDEC(BRB_DEC):
    """
    The BRB Improved Deep Embedded Clustering (IDEC) algorithm.
    Is equal to the DEC algorithm but uses the reconstruction loss also during the clustering optimization.
    Further, cluster_loss_weight is set to 0.1 instead of 1 when using the default settings.

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    labels: list
        list of ground truth labels
    checkpoint_path : str
        path to where model checkpoints should be saved (default: None)
    checkpointing_frequency : int
        frequency with which model checkpoints should be saved (default: 10)
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 0.1)
    data_reg_loss_weight : float
        weight of the regularization loss (reconstruction or contrastive loss) loss compared clustering loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    device : torch.device
            device to be trained on (default: None). If None, device will be set automatically using detect_device function.
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_cluster_centers : np.array
        initial cluster_centers to be used for DEC (default: None).
    reset_args: dict[str, int | float | bool]
        Various parameters for the reset procedure. See brb_reclustering for details.
    wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
        wandb run object for logging clustering training progress (default: None)
    log_interval: int
        Interval at which clustering metrics should be logged (default: 1)
    num_workers: int
        number of workers used for dataloader (default: 1)
    track_silhouette: bool
        If True, Inter-CD, Intra-DC and Silhouette score will be calculated and logged during training (default: False)
    track_purity: bool
        If True, local purity will be calculated and logged during training (default: False)
    track_voronoi: bool
        If True, Voronoi plots will be calculated and logged during training (default: False)
    track_uncertainty_plot: bool
        If True, uncertainty plot will be calculated and logged during training (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dc_labels_ : np.ndarray
        The final DEC labels
    dc_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder
    metrics : dict
        Dictionary with various performance metrics calculated during training
    reset_args: dict[str, int | float | bool]

    Examples
    ----------
    >>> from src.data import create_subspace_data
    >>> from src.deep import BRB_IDEC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50))
    >>> idec = BRB_IDEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
    >>> idec.fit(data)

    References
    ----------
    Guo, Xifeng, et al. "Improved deep embedded clustering with local structure preservation." IJCAI. 2017.
    """

    def __init__(
        self,
        n_clusters: int,
        checkpoint_path: str = None,
        checkpointing_frequency: int = 10,
        initial_cluster_centers: np.array = None,
        alpha: float = 1.0,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_optimizer_params: dict = {"lr": 1e-4},
        clustering_lr_scheduler_args: dict = {},
        pretrain_epochs: int = 100,
        clustering_epochs: int = 150,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: torch.nn.Module = None,
        embedding_size: int = 10,
        cluster_loss_weight: float = 0.1,
        data_reg_loss_weight: float = 1.0,
        custom_dataloaders: tuple = None,
        device: torch.device = None,
        augmentation_invariance: bool = False,
        reset_args: dict[str, int | float | bool] = {},
        wandb_run: Run | RunDisabled | None = None,
        log_interval: int = 1,
        num_workers: int = 1,
        track_silhouette: bool = False,
        track_purity: bool = False,
        track_voronoi: bool = False,
        track_uncertainty_plot: bool = False,
    ):
        super().__init__(
            n_clusters=n_clusters,
            checkpoint_path=checkpoint_path,
            checkpointing_frequency=checkpointing_frequency,
            initial_cluster_centers=initial_cluster_centers,
            alpha=alpha,
            batch_size=batch_size,
            pretrain_optimizer_params=pretrain_optimizer_params,
            clustering_optimizer_params=clustering_optimizer_params,
            clustering_lr_scheduler_args=clustering_lr_scheduler_args,
            pretrain_epochs=pretrain_epochs,
            clustering_epochs=clustering_epochs,
            optimizer_class=optimizer_class,
            loss_fn=loss_fn,
            autoencoder=autoencoder,
            embedding_size=embedding_size,
            cluster_loss_weight=cluster_loss_weight,
            data_reg_loss_weight=data_reg_loss_weight,
            custom_dataloaders=custom_dataloaders,
            device=device,
            augmentation_invariance=augmentation_invariance,
            reset_args=reset_args,
            wandb_run=wandb_run,
            log_interval=log_interval,
            num_workers=num_workers,
            track_silhouette=track_silhouette,
            track_purity=track_purity,
            track_voronoi=track_voronoi,
            track_uncertainty_plot=track_uncertainty_plot,
        )
        self.use_reconstruction_loss = True
