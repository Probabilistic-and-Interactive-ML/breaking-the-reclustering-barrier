import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

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
from src.deep.dcn_module import _BRB_DCN_Module
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ._clustering_utils import embedded_kmeans_prediction


class BRB_DCN(BaseEstimator, ClusterMixin):
    """
    The BRB Deep Clustering Network (DCN) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the AE will be optimized using the DCN loss function.

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    checkpoint_path : str
        path to where model checkpoints should be saved (default: None)
    checkpointing_frequency : int
        frequency with which model checkpoints should be saved (default: 10)
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
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 0.025)
    reconstruction_loss_weight : float
        weight of the reconstruction loss (default: 1.0)
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    device : torch.device
        device to be trained on (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_cluster_centers : np.array
        initial cluster_centers to be used for DCN (default: None).
    reset_args: dict[str, int | float | bool]
        Parameters for the resetting methods.
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
        If True, the voronoi diagram will be calculated and logged during training (default: False)
    track_uncertainty_plot: bool
        If True, the uncertainty plot will be calculated and logged during training (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dc_labels_ : np.ndarray
        The final DCN labels
    dc_cluster_centers_ : np.ndarray
        The final DCN cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder
    metrics : dict
        Dictionary with various performance metrics calculated during training
    wandb_run : wandb.sdk.lib.Run | wandb.wandb_run.RunDisabled | None
        wandb run object for logging clustering training progress (default: None)
    reset_args: dict[str, int | float | bool]

    References
    ----------
    Yang, Bo, et al. "Towards k-means-friendly spaces:
    Simultaneous deep learning and clustering." international
    conference on machine learning. PMLR, 2017.
    """

    def __init__(
        self,
        n_clusters: int,
        checkpoint_path: str = None,
        checkpointing_frequency: int = 10,
        initial_cluster_centers: np.array = None,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_optimizer_params: DCOptimizerArgs = DCOptimizerArgs(),
        pretrain_epochs: int = 100,
        clustering_epochs: int = 150,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: torch.nn.Module = None,
        embedding_size: int = 10,
        cluster_loss_weight: float = 0.025,
        representation_loss_weight: float = 1.0,
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
        self.n_clusters = n_clusters
        self.checkpoint_path = checkpoint_path
        self.checkpointing_frequency = checkpointing_frequency
        self.metrics = None
        self.initial_cluster_centers = initial_cluster_centers
        self.device = device
        self.batch_size = batch_size
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.representation_loss_weight = representation_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.num_workers = num_workers

        # BRB parameters
        self.reset_args = reset_args

        # wandb run object for logging clustering training progress
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.track_silhouette = track_silhouette
        self.track_purity = track_purity
        self.track_voronoi = track_voronoi
        self.track_uncertainty_plot = track_uncertainty_plot

    def fit(self, X: np.ndarray | None, y: np.ndarray) -> "BRB_DCN":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set. Can be None if custom_dataloaders is specified
        y : np.ndarray
            the labels

        Returns
        -------
        self : DCN
            this instance of the DCN algorithm
        """

        # Sanity checks
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

        # Setup DCN Module
        dcn_module = _BRB_DCN_Module(
            init_np_centers=init_centers,
            ground_truth_labels=y,
            augmentation_invariance=self.augmentation_invariance,
        ).to_device(self.device)
        optimizer = setup_optimizer(
            model=autoencoder,
            dc_module=None,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.clustering_optimizer_params,
            freeze_convlayers=self.clustering_optimizer_params.freeze_convlayers,
        )

        # DCN Training loop
        dcn_module.fit(
            autoencoder=autoencoder,
            trainloader=trainloader,
            trainloader_wA=trainloader_wA,
            testloader=testloader,
            n_epochs=self.clustering_epochs,
            device=self.device,
            optimizer=optimizer,
            lr_scheduler_args=self.clustering_optimizer_params,
            loss_fn=self.loss_fn,
            cluster_loss_weight=self.cluster_loss_weight,
            reconstruction_loss_weight=self.representation_loss_weight,
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
        self.dc_labels_ = predict_batchwise(trainloader_wA, autoencoder, dcn_module, self.device)
        self.dc_cluster_centers_ = dcn_module.centers.detach().cpu().numpy()
        self.metrics = dcn_module.metrics

        # Do reclustering with Kmeans
        embedded_data = encode_batchwise(trainloader_wA, autoencoder, self.device)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init="auto")
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
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, self.autoencoder)
        return predicted_labels
