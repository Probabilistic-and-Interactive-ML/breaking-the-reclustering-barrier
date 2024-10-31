from dataclasses import dataclass
from typing import Literal

from config.types import RunnerBooleanTupleArg, SchedulerArg


@dataclass
class BaseOptimizerArgs:
    """Base configuration for the optimizer."""

    # Optimizer
    optimizer: Literal["adam", "sgd"] | tuple[Literal["adam", "sgd"], ...] = "adam"
    # Learning rate
    lr: float | tuple[float, ...] = 1e-3


@dataclass
class PretrainOptimizerArgs(BaseOptimizerArgs):
    """Configuration for the pretraining optimizer."""

    # Weight decay
    weight_decay: float | tuple[float, ...] = 0.0
    # Weight decay should be smaller then whole network weight decay or 0
    projector_weight_decay: float | tuple[float, ...] = 0.0
    # Learning rate. Initial lr if scheduling is used
    lr: float | tuple[float, ...] = 1e-4
    # Whether to schedule the learning rate for the autoencoder
    schedule_lr: RunnerBooleanTupleArg = False


@dataclass
class DCOptimizerArgs(BaseOptimizerArgs):
    """Configuration for the clustering optimizer."""

    # Weight decay
    weight_decay: float | tuple[float, ...] = 0.0
    # Weight decay should be smaller then whole network weight decay or 0
    projector_weight_decay: float | tuple[float, ...] = 0.0
    # Whether to freeze convolutional part of the Resnet
    freeze_convlayers: RunnerBooleanTupleArg = False
    # Learning rate. Initial lr if scheduling is used
    lr: float | tuple[float, ...] = 1e-3

    # Whether to schedule the learning rate for the clustering algorithm and which method to use
    scheduler_name: SchedulerArg | tuple[SchedulerArg, ...] = "none"
    # Step size for the learning rate scheduler
    scheduler_stepsize: int | tuple[int, ...] = 50
    # Gamma for the learning rate scheduler
    scheduler_gamma: float | tuple[float, ...] = 0.5
    # T_0 for the cosine_warm_restarts learning rate scheduler
    scheduler_T_0: int | tuple[int, ...] = 20
    # T_mult for the cosine_warm_restarts learning rate scheduler
    scheduler_T_mult: int | tuple[int, ...] = 1
    # Total epochs for the cosine_warm_restarts learning rate scheduler
    scheduler_total_epochs: int | tuple[int, ...] = 20
    # Warmup epochs for the linear_warmup_cosine learning rate scheduler
    scheduler_warmup_epochs: int | tuple[int, ...] = 5
    # Start factor for the linear_warmup_cosine learning rate scheduler
    scheduler_start_factor: float | tuple[float, ...] = 0.1
    # End factor for the linear_warmup_cosine learning rate scheduler
    scheduler_end_factor: float | tuple[float, ...] = 1.0
    # Reset lr scheduler after BRB reset
    reset_lr_scheduler: RunnerBooleanTupleArg = False
