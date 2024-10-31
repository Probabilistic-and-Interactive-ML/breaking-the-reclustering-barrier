from dataclasses import dataclass

from tyro.conf import FlagConversionOff

from config.types import ReClusteringArg, RunnerBooleanTupleArg


@dataclass
class BRBArgs:
    # BRB settings: reset_weights=True, recluster=True, recalculate_centers=False, reset_momentum=True
    # Reclustering only settings: reset_weights=False, recluster=True, recalculate_centers=False, reset_momentum=False
    # Whether to apply a weight reset or not
    reset_weights: RunnerBooleanTupleArg = True
    # Whether to recluster or not
    recluster: RunnerBooleanTupleArg = True
    # If True, centers will be recalculated using the previous
    # predicted cluster labels and the new features after the reset
    recalculate_centers: RunnerBooleanTupleArg = False
    # Reset momentum for DEC/IDEC centroids if adam optimizer is used
    reset_momentum: RunnerBooleanTupleArg = True
    # Reset momentum counts
    reset_momentum_counts: RunnerBooleanTupleArg = False
    # Interpolation factor for soft reset. 0 is a hard reset, 1 is no reset
    reset_interpolation_factor: float | tuple[float, ...] = 0.7
    # Epochs after which to perform resets for BRB
    reset_interval: int | tuple[int, ...] = 2
    # Reset BatchNorm affine parameters
    reset_batchnorm: RunnerBooleanTupleArg = False
    # Wether to use eval mode for batchnorm during clustering or not
    batchnorm_eval_mode: RunnerBooleanTupleArg = False
    # Reset embedding
    reset_embedding: RunnerBooleanTupleArg = False
    # Reset contrastive projector
    reset_projector: RunnerBooleanTupleArg = False
    # Reset convolutional layers
    # If True then all convolutional blocks of the resnet will be reset
    # Alternatively a string of comma separated integers can be passed to indicate which resnet blocks should be reset.
    # e.g., resnet18 has 4 convlayer blocks, so the string "4,3,2" will reset the 4th, 3rd and 2nd block respectively, where
    # the 4th block is the deepest block, right before the pooling operation.
    reset_convlayers: FlagConversionOff[bool] | str | tuple[str, ...] = False
    # Step size for the interpolation factor
    # e.g. if reset_interpolation_factor=0.8, reset_interpolation_factor_step=0.05 and convlayers "1,2,3,4" are reset:
    # then the mlp head will be reset with 0.8, convlayer 4 with 0.85, 3 with 0.9, 2 with 0.95 and convlayer 1 with 1.0 (no reset)
    # the reset_interpolation_factor is capped at 1.0 (no reset), so if your reset_interpolation_factor is 0.9 in the example above,
    # then the mlp head will be reset with 0.9, convlayer 4 with 0.95, 3 with 1.0 (no reset), 2 with 1.0 (no reset) and convlayer 1 with 1.0 (no reset) as well
    # in this case a warning is printed that the layers will not be reset
    reset_interpolation_factor_step: float | tuple[float, ...] = 0.05
    # Determines which method should be used for reclustering
    # kmeans: Performs full kmeans clustering on the embedding at epoch X
    # kmeans++-init: Performs the kmeans++-init init on the embedding at epoch X
    # kmedoids: Performs kmedoids clustering on the embedding at epoch X
    # em: Performs expectation maximization clustering to fit a GMM on the embedding at epoch X
    # random: samples random cluster centers from the embedding at epoch X
    reclustering_method: ReClusteringArg | tuple[ReClusteringArg, ...] = "kmeans"
    # Specify size of subsample that should be used for reclustering.
    # If None, than full data will be used for r
    subsample_size: int | None | tuple[int | None] = 10000

    def __post_init__(self) -> None:
        # Handle erroneously parsed boolean arguments
        if self.reset_convlayers == ("True",):
            self.reset_convlayers = True
        if self.reset_convlayers == ("False",):
            self.reset_convlayers = False
