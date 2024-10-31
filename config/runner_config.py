from dataclasses import dataclass
from typing import Literal

from tyro.conf import FlagConversionOff

from config.base_config import (
    BRBArgs,
    DCOptimizerArgs,
    ExperimentArgs,
    PretrainOptimizerArgs,
    _set_debug_params,
    _set_irrelevant_reset_params_const,
)
from config.types import (
    ClusterArg,
    ConvnetArg,
    DataSetArg,
    DataSubsetArg,
    ModelArg,
    RunnerBooleanTupleArg,
)


@dataclass
class RunnerArgs:
    """Configuration for a batch of runs.

    This is mostly the copy-pasted version of src/config.py allowing for tuple arguments in sensible places.
    Correct tuple typing is important for the CLI to work.

    - Setting --debug=True performs a dry run with minimal settings and no logging.
    - Setting --track_wandb=True tracks the experiment on wandb.
    - Settings for a reset method that are irrelevant will automatically get set to 0.

    It does not iterate over the following arguments even when they're tuples:
    - gpu
    - num_workers
    """

    # Hardware settings
    experiment: ExperimentArgs

    # Optimizer settings
    pretrain_optimizer: PretrainOptimizerArgs
    dc_optimizer: DCOptimizerArgs

    # Reset parameters
    brb: BRBArgs

    # Runner parameters
    command: str = "python train.py"
    workers: int = 4

    seed: int | tuple[int, ...] = (400, 148, 820, 214, 40, 805, 111, 491, 215, 513)

    # mini-batch size
    batch_size: int | tuple[int, ...] = 256

    # Dataset parameters
    dataset_name: DataSetArg | tuple[DataSetArg, ...] = (
        "cifar10",
        "mnist",
        "optdigits",
        "usps",
        "fmnist",
        "kmnist",
        "gtsrb",
        "cifar100_20",
    )
    dataset_subset: DataSubsetArg = "all"

    # AE parameters
    model_type: ModelArg | tuple[ModelArg, ...] = (
        "feedforward_small",
        "feedforward_medium",
        "feedforward_large",
        "feedforward_less_depth",
    )
    # Specify convnet to be used. If "none" than only feed forward AE will be used
    convnet: ConvnetArg | tuple[ConvnetArg, ...] = "none"
    # Whether to load a pretrained autoencoder if possible: overwrite_ae=True means a fresh one will always be trained.
    overwrite_ae: FlagConversionOff[bool] = True
    # Where to load the AE model from
    ae_path: str | None = None
    # Whether to save the autoencoder after training
    save_ae: FlagConversionOff[bool] = False
    # Number of pretraining epochs. If 0 pretraining will be skipped
    pretrain_epochs: int | tuple[int, ...] = 250
    # Activation function
    activation_fn: Literal["leaky_relu", "relu"] | tuple[Literal["leaky_relu", "relu"], ...] = ("relu", "leaky_relu")
    # BatchNorm
    batch_norm: RunnerBooleanTupleArg = True
    # Embedding dimensionality
    embedding_dim: int | None | tuple[int | None, ...] = None
    # Normalize the embeddings in the encoder
    normalize_embeddings: RunnerBooleanTupleArg = False
    # If true, pretrain with augmentations.
    # NOTE: This will be set to True automatically if use_contrastive_loss is True, as we need augmentations for contrastive learning
    augmented_pretraining: RunnerBooleanTupleArg = True
    # Convnet specific args
    # Whether to use a last linear layer in the convnet
    additional_last_linear_layer: RunnerBooleanTupleArg = False

    # Data loading parameters
    # If true than, than the augmentation dataloader will generate two augmented samples,
    # instead of only one (while keeping the second one unaugmented)
    # NOTE: Setting this to true should lead to better performance with contrastive learning
    augment_both_views: RunnerBooleanTupleArg = False

    # Contrastive Learning Parameters for AE
    # If set to true, contrastive loss will be used instead of reconstruction loss.
    # NOTE: Specified parameters below are only applied if use_contrastive_loss is set to True.
    use_contrastive_loss: FlagConversionOff[bool] = False
    # Temperature parameter of the softmax used in the contrastive loss (smaller tau leads to "harder" assignments)
    softmax_temperature_tau: float | tuple[float, ...] = 0.1
    # specify number of layers for projector MLP, if 0 no projector will be used
    projector_depth: int | tuple[int, ...] = 0
    # Specify the size of the projector layer, e.g, with projector depth of 2 and projector_layer_size of 128,
    # we would pass the following layers to the network: [embedding_dim, 128, 128, embedding_dim].
    # If None, then size of embedding_dim will be used.
    # NOTE: This will only be used if projector_depth > 0.
    projector_layer_size: int | None | tuple[int, ...] = None
    # If set to True, projector will be added to output of resnet instead of the output of the MLP cluster head.
    # NOTE: Works currently only for Resnet architecture
    separate_cluster_head: RunnerBooleanTupleArg = False

    # Clustering parameters
    # Clustering algorithm to use
    dc_algorithm: ClusterArg | tuple[ClusterArg, ...] = ("dcn", "idec", "dec")
    # Weight for the clustering loss
    cluster_loss_weight: float | None | tuple[float | None, ...] = None
    # Weight for data-dependent regularization loss (either reconstruction or contrastive loss). Ignored for DEC
    data_reg_loss_weight: float | tuple[float, ...] = 1.0

    # Number of epochs for the clustering algorithm. If 0 clustering will be skipped
    clustering_epochs: int | tuple[int, ...] = 200
    # Flag for saving the clustering model
    save_clustering_model: FlagConversionOff[bool] = False
    # Whether to load a pretrained clustering model if possible
    load_clustering_model: FlagConversionOff[bool] = False
    # Number of clusters to use, if None then ground truth number of clusters is specified automatically from the labels
    n_clusters: int | None | tuple[int, ...] = None
    # Whether to apply augmentations during clustering to learn cluster assignments that are invariant to the augmentations specified in src/datasets/augmentation_dataloader.py
    augmentation_invariance: RunnerBooleanTupleArg = True
    # Crop size used for torchvision.transforms.RandomResizedCrop. This is only used for color images
    crop_size: int | tuple[int, ...] = 32

    def __post_init__(self):
        """Set debug parameters if debug mode is enabled."""

        if self.experiment.debug:
            _set_debug_params(self)

        _set_irrelevant_reset_params_const(self)


# NOTE: Don't have to set all settings because we have the defaults in the ExperimentArgs above
Configs = {
    ######################################################################
    # DEBUG and Test Runs
    ######################################################################
    "test": RunnerArgs(
        command="python train.py",
        workers=6,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - Test"',
            result_dir=None,
            prefix="test",
            tag=("test"),
            gpu=(0, 1),
            track_wandb=True,
            debug=True,
            wandb_entity=None,
        ),
        pretrain_optimizer=PretrainOptimizerArgs(
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        dc_optimizer=DCOptimizerArgs(
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            freeze_convlayers=False,
            scheduler_name="none",
        ),
        brb=BRBArgs(
            reset_weights=False,
            reset_interpolation_factor=0.5,
        ),
        seed=0,  # , 805, 111, 491, 215, 513),
        dataset_name=("usps", "gtsrb"),
        model_type="feedforward_large",
        use_contrastive_loss=False,
        activation_fn="relu",
        embedding_dim=None,  # 5, 50, 100
        dc_algorithm="idec",
    ),
    "default": RunnerArgs(
        command="python train.py",
        experiment=ExperimentArgs(),
        pretrain_optimizer=PretrainOptimizerArgs(),
        dc_optimizer=DCOptimizerArgs(),
        brb=BRBArgs(),
    ),
}
