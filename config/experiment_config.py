from dataclasses import dataclass
from typing import Literal

from tyro.conf import FlagConversionOff


@dataclass
class ExperimentArgs:
    """Configuration for hardware and logging."""

    num_workers_dataloader: int = 20
    gpu: int | None | Literal["all"] | tuple[int, ...] = "all"
    deterministic_torch: FlagConversionOff[bool] = True
    debug: FlagConversionOff[bool] = False

    # General experiment settings
    wandb_project: str = "Test"
    prefix: str = "test"
    tag: str | tuple[str, ...] | None = "test"
    # GPU or GPUs to run on
    # int: Run on a single GPU
    # tuple[int]: Run on multiple GPUs
    # "all": Run on all available GPUs
    # None: Run on CPU
    track_wandb: FlagConversionOff[bool] = False
    # log to personal account as default by setting to None
    wandb_entity: str | None = None
    wandb_check_duplicates: FlagConversionOff[bool] = False
    # Number of workers for dataloader
    result_dir: str | None = None  # "/mnt/data/miklautzl92dm_data/dc_plasticity/exploration"
    # Optional computation for expensive metrics
    # Track Silhouette score
    track_silhouette: FlagConversionOff[bool] = False
    # Track local purity
    track_purity: FlagConversionOff[bool] = False
    # Intervals in which clustering metrics are computed and logged
    cluster_log_interval: int = 1
    # Voronoi diagram
    track_voronoi: FlagConversionOff[bool] = False
    # Uncertainty Score Plot
    track_uncertainty_plot: FlagConversionOff[bool] = False
