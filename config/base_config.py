from dataclasses import dataclass
from typing import Literal

from tyro.conf import FlagConversionOff

from config.brb_config import BRBArgs
from config.experiment_config import ExperimentArgs
from config.optimizer_config import DCOptimizerArgs, PretrainOptimizerArgs
from config.types import ClusterArg, ConvnetArg, DataSetArg, DataSubsetArg, ModelArg


@dataclass
class Args:
    """Configuration for training.

    - Setting --debug=True performs a dry run with minimal settings and no logging.
    - Setting --track_wandb=True tracks the experiment on wandb.
    - Settings for a reset method that are irrelevant will automatically get set to 0.

    """

    # Hardware settings
    experiment: ExperimentArgs

    # Optimizer settings
    pretrain_optimizer: PretrainOptimizerArgs
    dc_optimizer: DCOptimizerArgs

    # Reset parameters
    brb: BRBArgs

    seed: int = 123

    # mini-batch size
    batch_size: int = 512

    # Dataset parameters
    dataset_name: DataSetArg = "usps"
    dataset_subset: DataSubsetArg = "test"

    # Feed forward AE parameters
    model_type: ModelArg = "feedforward_large"
    # Specify convnet to be used. If "none" than only feed forward AE will be used
    convnet: ConvnetArg = "none"
    # Whether to load a pretrained autoencoder if possible: overwrite_ae=True means a fresh one will always be trained.
    overwrite_ae: FlagConversionOff[bool] = False
    # Where to load the AE model from
    ae_path: str | None = None
    # Whether to save the autoencoder after training
    save_ae: FlagConversionOff[bool] = True
    # Number of pretraining epochs. If 0 pretraining will be skipped
    pretrain_epochs: int = 10  # #5#0  # 50
    # Activation function
    activation_fn: Literal["leaky_relu", "relu"] = "relu"
    # BatchNorm
    batch_norm: FlagConversionOff[bool] = False
    # Embedding dimensionality
    embedding_dim: int | None = 256
    # Normalize the embeddings in the encoder
    normalize_embeddings: FlagConversionOff[bool] = False
    # If true, pretrain with augmentations.
    # NOTE: This will be set to True automatically if use_contrastive_loss is True, as we need augmentations for contrastive learning
    augmented_pretraining: FlagConversionOff[bool] = False
    # Convnet specific args
    # Whether to use a last linear layer in the convnet
    additional_last_linear_layer: FlagConversionOff[bool] = False

    # Data loading parameters
    # If true than, than the augmentation dataloader will generate two augmented samples,
    # instead of only one (while keeping the second one unaugmented)
    # NOTE: Setting this to true should lead to better performance with contrastive learning
    augment_both_views: FlagConversionOff[bool] = False

    # Contrastive Learning Parameters for AE
    # If set to true, contrastive loss will be used instead of reconstruction loss.
    # NOTE: Specified parameters below are only applied if use_contrastive_loss is set to True.
    use_contrastive_loss: FlagConversionOff[bool] = False
    # Temperature parameter of the softmax used in the contrastive loss (smaller tau leads to "harder" assignments)
    softmax_temperature_tau: float = 0.5
    # specify number of layers for projector MLP, if 0 no projector will be used
    projector_depth: int = 1
    # Specify the size of the projector layer, e.g, with projector depth of 2 and projector_layer_size of 128,
    # we would pass the following layers to the network: [embedding_dim, 128, 128, embedding_dim].
    # If None, then size of embedding_dim will be used.
    # NOTE: This will only be used if projector_depth > 0.
    projector_layer_size: int | None = 2048
    # If set to True, projector will be added to output of resnet instead of the output of the MLP cluster head.
    # NOTE: Works currently only for Resnet architecture
    separate_cluster_head: FlagConversionOff[bool] = False

    # Clustering parameters
    # Clustering algorithm to use
    dc_algorithm: ClusterArg = "dcn"
    # Weight for the clustering loss
    cluster_loss_weight: float | None = None
    # Weight for data-dependent regularization loss (either reconstruction or contrastive loss). Ignored for DEC
    data_reg_loss_weight: float = 1.0

    # Number of epochs for the clustering algorithm. If 0 clustering will be skipped
    clustering_epochs: int = 10
    # Flag for saving the clustering model
    save_clustering_model: FlagConversionOff[bool] = False
    # Whether to load a pretrained clustering model if possible
    load_clustering_model: FlagConversionOff[bool] = False
    # Number of clusters to use, if None then ground truth number of clusters is specified automatically from the labels
    n_clusters: int | None = None
    # Whether to apply augmentations during clustering to learn cluster assignments that are invariant to the augmentations specified in src/datasets/augmentation_dataloader.py
    augmentation_invariance: FlagConversionOff[bool] = True
    # Crop size used for torchvision.transforms.RandomResizedCrop. This is only used for color images
    crop_size: int = 32

    def __post_init__(self):
        """Set debug parameters if debug mode is enabled."""

        if isinstance(self.brb.reset_interval, tuple) and len(self.brb.reset_interval) == 1:
            self.brb.reset_interval = self.brb.reset_interval[0]

        if self.experiment.debug:
            _set_debug_params(self)

        if self.cluster_loss_weight is None:
            # 0.025 is the default for DCN, 0.1 for IDEC. For DEC, it doesn't matter
            if self.dc_algorithm == "dcn":
                self.cluster_loss_weight = 0.025
            elif self.dc_algorithm == "idec":
                self.cluster_loss_weight = 0.1
            elif self.dc_algorithm == "dec":
                self.cluster_loss_weight = 0.0

        # Only call at the end of post_init when everything else is done.
        _set_irrelevant_reset_params_const(self)

        # Check that the reset interpolation factor is valid when BRB is used
        if self.brb.reset_weights:
            if not 0.0 <= self.brb.reset_interpolation_factor <= 1:
                raise ValueError(
                    f"Interpolation parameter is expected to be between 0.0 and 1.0, but was: {self.brb.reset_interpolation_factor}."
                )
        # Check that brb_reset_interval is valid, else the check for resets will fail
        assert self.brb.reset_interval > 0, "Reset interval must be positive."


def _set_debug_params(args) -> None:
    """Sets the debug parameters for the given arguments class IN-PLACE."""
    # Deactivated because the warnings clutter the output
    args.experiment.deterministic_torch = False
    args.experiment.track_wandb = False

    args.overwrite_ae = True
    args.save_ae = False

    args.pretrain_epochs = 1
    args.clustering_epochs = 1

    args.save_clustering_model = False
    args.load_clustering_model = False


def _set_irrelevant_reset_params_const(args) -> None:
    """Sets all reset parameters that are irrelevant for a specific method to a constant \" off \"value.

    NOTE that this modifies the arguments class in-place.
    """

    if args.dc_algorithm == "dcn":
        # DCN centroids are not learnable parameters
        args.brb.reset__momentum = False
