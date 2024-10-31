import copy
import subprocess
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tyro
from tqdm import tqdm

import wandb

NumType = float | int

# List of all relevant hyperparameter settings that we might want to use for filtering
HYPERPARAMS = {
    "experiment.seed",
    "dataset_name",
    "dataset_subset",
    "model_type",
    "convnet",
    "batch_size",
    "pretrain_epochs",
    "pretrain_optimizer.weight_decay",
    "pretrain_optimizer.lr",
    "pretrain_optimizer.schedule_lr",
    "pretrain_optimizer.projector_weight_decay",
    "pretrain_optimizer.freeze_convlayers",
    "dc_optimizer.weight_decay",
    "dc_optimizer.projector_weight_decay",
    "dc_optimizer.freeze_convlayers",
    "dc_optimizer.lr",
    "dc_optimizer.scheduler_name",
    "dc_optimizer.scheduler_stepsize",
    "dc_optimizer.scheduler_gamma",
    "dc_optimizer.scheduler_T_0",
    "dc_optimizer.scheduler_T_mult",
    "dc_optimizer.scheduler_total_epochs",
    "dc_optimizer.scheduler_warmup_epochs",
    "dc_optimizer.scheduler_start_factor",
    "dc_optimizer.scheduler_end_factor",
    "dc_optimizer.reset_lr_scheduler",
    "activation_fn",
    "batch_norm",
    "optimizer",
    "embedding_dim",
    "augmented_pretraining",
    "augment_both_views",
    "use_contrastive_loss",
    "use_additional_reconstruction_loss",
    "softmax_temperature_tau",
    "projector_depth",
    "projector_layer_size",
    "dc_algorithm",
    "cluster_loss_weight",
    "data_reg_loss_weight",
    "clustering_epochs",
    "n_clusters",
    "augmentation_invariance",
    "crop_size",
    "brb.reset_weights",
    "brb.recluster",
    "brb.recalculate_centers",
    "brb.reset_momentum",
    "brb.reset_momentum_counts",
    "brb.reset_interpolation_factor",
    "brb.reset_interpolation_factor_step",
    "brb.reset_interval",
    "brb.reset_embedding",
    "brb.reset_projector",
    "brb.reset_convlayers",
    "brb.reset_batchnorm",
    "brb.reclustering_method",
    "brb.subsample_size",
}


@dataclass
class DownloadArgs:
    # The wandb account/group to download from
    entity: str | None = None
    # The name of the wandb project
    project: str | None = "Test"
    sweep_id: str | None = None
    is_sweep: bool = False

    # AE metrics to extract
    ae_metrics: str | tuple[str, ...] | None = None  # (
    #     "reconstruction_loss",
    #     "l0_grad_norm",
    #     "l1_grad_norm",
    #     "l2_grad_norm",
    # )

    # Clustering train metrics to extract
    cluster_metrics: str | tuple[str, ...] | None = None  # (
    #     "cluster_loss",
    #     "l1_grad_norm",
    #     "l2_grad_norm",
    #     "l0_grad_norm",
    #     "AE_loss",
    #     "cluster_loss",
    # )

    # Clustering performance metrics to extract
    cluster_performance_metrics: str | tuple[str, ...] | None = (
        "AE_contrastive_loss",
        "AE_reconstruction_loss",
        "ACC",
        "ARI",
        "NMI",
        # "SIL",
        # "LocalPurity100",
        # "LocalPurity10",
        # "LocalPurity1",
        # "LocalPurity_MinClusterSize",
        # "Inter_CD",
        # "Intra_CD",
    )

    def __post_init__(self) -> None:
        # Prepend the wandb panel group
        cluster_perf_metrics = copy.deepcopy(self.cluster_performance_metrics)
        self.ae_metrics = tuple(f"AE train/{metric}" for metric in self.ae_metrics) if self.ae_metrics is not None else None
        self.cluster_metrics = (
            tuple(f"Clustering train/{metric}" for metric in self.cluster_metrics)
            if self.cluster_metrics is not None
            else None
        )
        self.cluster_performance_metrics = (
            tuple(f"Clustering metrics/{metric}" for metric in cluster_perf_metrics)
            if cluster_perf_metrics is not None
            else None
        )
        self.cluster_test_metrics = (
            tuple(f"Clustering test metrics/{metric}" for metric in cluster_perf_metrics)
            if cluster_perf_metrics is not None
            else None
        )

        if self.entity is None:
            # Auto-set the entity to the logged in user
            proc = subprocess.run(["wandb login"], shell=True, text=True, capture_output=True)
            self.entity = proc.stderr.split(".")[0].split(" ")[-1]

        assert self.project is not None
        if self.is_sweep:
            assert self.sweep_id is not None


def _sanity_checks(ae_metrics: dict[str, NumType], cluster_performance: dict[str, NumType], ae_trained: bool) -> None:
    """Some sanity checks to ensure that the data is correct and help with debugging."""
    if ae_trained:
        for k, v in ae_metrics.items():
            assert len(v) == len(ae_metrics["Step"]), f"Length of {k} is {len(v)} but should be {len(ae_metrics['Step'])}"
            if v.count(None) == len(v):
                warnings.warn(f"Column {k} is all None. Consider whether this is intended.")

    for k, v in cluster_performance.items():
        assert len(v) == len(
            cluster_performance["Step"]
        ), f"Length of {k} is {len(v)} but should be {len(cluster_performance['Step'])}"
        if v.count(None) == len(v):
            warnings.warn(f"Column {k} is all None. Consider whether this is intended.")


def _pad_cluster_metrics(cluster_performance: dict[str, NumType]) -> dict[str, NumType]:
    """Pad the cluster metrics with NaNs to ensure that all metrics have the same length."""

    # Hacky: Get the length of the longest list that is a dictionary value and add a step index
    max_len = len(max(cluster_performance.items(), key=lambda x: len(x[1]))[1])
    cluster_performance["Step"] = list(range(max_len))

    # Ensure that all metrics have the same length for later pandas operations
    for k, v in cluster_performance.items():
        if len(v) == max_len - 1:
            # Training metrics have one less step than performance metrics because there is no initial evaluation
            v.insert(0, float("nan"))
        elif len(v) < max_len - 1:
            # Something fucky
            raise ValueError(f"Length of {k} is {len(v)} but should either be {max_len} or {max_len - 1}.")
        else:
            # Length is max_len. We're good
            pass

    return cluster_performance


def _add_hyperparams_to_results(
    run,
    run_id: int,
    ae_metrics: dict[str, NumType],
    cluster_performance: dict[str, NumType],
    cluster_test_metrics: dict[str, NumType] | None,
    ae_trained: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add core information to all the dictionaries for later filtering."""

    ae_metrics["ID"] = [run_id] * len(ae_metrics["Step"])
    ae_metrics["tags"] = [run.tags] * len(ae_metrics["Step"])
    cluster_performance["ID"] = [run_id] * len(cluster_performance["Step"])
    cluster_performance["tags"] = [run.tags] * len(cluster_performance["Step"])

    # Cluster test metrics can be None for old runs
    if cluster_test_metrics is not None:
        cluster_test_metrics["ID"] = run_id
        cluster_test_metrics["tags"] = run.tags

    for key in HYPERPARAMS:
        # First checking whether the key exists in the config
        # makes the downloader backwards compatible (before these params were added)
        if key in run.config.keys():
            if ae_trained:
                ae_metrics[key] = [run.config[key]] * len(ae_metrics["Step"])

            cluster_performance[key] = [run.config[key]] * len(cluster_performance["Step"])

            if cluster_test_metrics is not None:
                cluster_test_metrics[key] = run.config[key]

    return ae_metrics, cluster_performance, cluster_test_metrics


def _fetch_run_results(
    run_history, args: DownloadArgs
) -> tuple[dict[str, NumType], dict[str, NumType], dict[str, NumType] | None]:
    """Fetch relevant metrics for a single run from the wandb history."""

    # Get important quantities and set NaNs to None for later filtering
    # Very hacky to just iterate once over the data

    all_metrics = []
    for m in [args.ae_metrics, args.cluster_metrics, args.cluster_performance_metrics]:
        if m is not None:
            all_metrics.extend(m)

    assert all_metrics, "No metrics to fetch. Please specify metrics in the DownloadArgs."

    raw_results = [{"Step": row.get("_step")} | {cmd: row.get(cmd, None) for cmd in all_metrics} for row in run_history]

    # Filter out the runs where training failed but that are still marked as finished in wandb
    if not raw_results:
        return None, None

    # Create a dictionary with the metrics for each type due to different step numbers
    ae_metrics = defaultdict(list)
    cluster_performance = defaultdict(list)
    for result in raw_results:
        for key, value in result.items():
            if value is not None:
                if key.startswith("AE"):
                    ae_metrics[key].append(value)
                elif key.startswith("Clustering metrics") or key.startswith("Clustering train"):
                    cluster_performance[key].append(value)

    # Fetch test metrics manually
    cluster_test_metrics = {
        k: v for k, v in raw_results[-1].items() if k.startswith("Clustering test metrics") and v is not None
    }

    # Set cluster test metrics to None if there are no test metrics
    if not cluster_test_metrics:
        cluster_test_metrics = None

    return ae_metrics, cluster_performance, cluster_test_metrics


def get_run_information(
    run, run_id: int, args: DownloadArgs
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch the relevant quantities for a single run."""

    # Setting a large page size is key for avoiding many queries to the API
    run_history = run.scan_history(page_size=100_000)

    # Fetch relevant metrics
    ae_metrics, cluster_performance, cluster_test_metrics = _fetch_run_results(run_history=run_history, args=args)

    # Add steps to all the dictionaries
    if ae_trained := bool(ae_metrics):
        ae_metrics["Step"] = list(range(len(next(iter(ae_metrics.values())))))

    # Pad cluster metrics with NaNs to ensure that all metrics have the same length
    cluster_performance = _pad_cluster_metrics(cluster_performance=cluster_performance)

    # Debugging
    _sanity_checks(ae_metrics=ae_metrics, cluster_performance=cluster_performance, ae_trained=ae_trained)

    # Add hyperparameters to the result DataFrames for later filtering
    ae_metrics, cluster_performance, cluster_test_metrics = _add_hyperparams_to_results(
        run=run,
        run_id=run_id,
        ae_metrics=ae_metrics,
        cluster_performance=cluster_performance,
        cluster_test_metrics=cluster_test_metrics,
        ae_trained=ae_trained,
    )

    # Return pandas dataframes
    if ae_trained:
        ae_metrics = pd.DataFrame(ae_metrics)
    else:
        ae_metrics = None

    if cluster_test_metrics is not None:
        cluster_test_metrics = pd.DataFrame(cluster_test_metrics, index=["value"])

    cluster_performance = pd.DataFrame(cluster_performance)

    return ae_metrics, cluster_performance, cluster_test_metrics


FILTER_PARAMS = {"tags", "State", "$or", "id"}
""" Additional parameters that can be used for filtering runs. """

# Copied from https://github.com/wandb/wandb/blob/c6b7a82a152e8efee84cb7f0887532c482ccd6c1/wandb/apis/public.py#L867-L902
"""
Examples:
    Find runs in my_project where config.experiment_name has been set to "foo":
    ```
    api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
    ```

    Find runs in my_project where config.experiment_name has been set to "foo" or "bar":
    ```
    api.runs(
        path="my_entity/my_project",
        filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
    )
    ```

    Find runs in my_project where config.experiment_name matches a regex (anchors are not supported):
    ```
    api.runs(
        path="my_entity/my_project",
        filters={"config.experiment_name": {"$regex": "b.*"}}
    )
    ```

    Find runs in my_project where the run name matches a regex (anchors are not supported):
    ```
    api.runs(path="my_entity/my_project", filters={"display_name": {"$regex": "^foo.*"}})
    ```

    Find runs in my_project sorted by ascending loss:
    ```
    api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
    ```
"""
FILTERS = {
    # Always leave this enabled
    # "State": "finished",
    # NOTE: Some example tags for attributes from the config
    # "config.dc_algorithm": "idec",
    # "config.clustering_lr": 0.001,
    # "config.model_type": "feedforward_small",
    # "config.seed": 400,
    # "config.dataset_name": "mnist"
    # NOTE: Project name needs to be adapted for each tag
    # Tag for baseline_experiments  --> Project Name: "Clustering BRB - Baseline"
    # "tags": "baseline_small2",
    # OR filter for several files
    # "$or": [{"config.dataset_name": "mnist"},
    #         {"config.dataset_name": "optdigits"},
    #         {"config.dataset_name": "cifar10"},
    #         ],
    # "config.dataset_name": "cifar10",
    # "id": "ovqtnwt9"
}

if __name__ == "__main__":
    args = tyro.cli(DownloadArgs)

    # Sanity check
    if FILTERS is not None:
        for key in FILTERS:
            assert key.split(".")[-1] in HYPERPARAMS | FILTER_PARAMS, f"Filter key {key} is not a valid hyperparameter."

    api = wandb.Api(timeout=60)
    if args.is_sweep:
        runs = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")
    else:
        runs = api.runs(f"{args.entity}/{args.project}", filters=FILTERS)

    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(__file__).parents[0] / "results" / f"{args.entity}" / f"{timestamp}_{args.project.replace(' ', '')}"
    save_path.mkdir(exist_ok=True, parents=True)

    ae_runs = []
    cluster_runs = []
    cluster_test = []

    pbar = tqdm(runs, desc="Downloading runs")
    for run_id, run in enumerate(pbar):
        ae_metrics, cluster_performance, cluster_test_metrics = get_run_information(run, run_id, args)
        ae_runs.append(ae_metrics)
        cluster_runs.append(cluster_performance)
        cluster_test.append(cluster_test_metrics)

        if cluster_performance is not None:
            pbar.set_postfix_str(
                f"ACC: {(100*cluster_performance['Clustering metrics/ACC'].iloc[-1]):.2f}"
                f" | ARI: {(100*cluster_performance['Clustering metrics/ARI'].iloc[-1]):.2f}"
            )

    # Concatenate all the runs. They can be identified by their run ID
    # For AE metrics, filter out the runs where no AE was trained. If no AE was trained, don't save anything
    if not all([r is None for r in ae_runs]):
        ae_runs = pd.concat([r for r in ae_runs if r is not None])
        ae_runs.to_pickle(save_path / "pretrain_metrics.pkl", compression="gzip")

    # Filter runs where no test metrics were logged
    if not all([r is None for r in cluster_test]):
        cluster_test = pd.concat([r for r in cluster_test if r is not None])
        cluster_test.to_pickle(save_path / "clustering_test_metrics.pkl", compression="gzip")

    # Filter runs where no data was uploaded
    cluster_runs = pd.concat([r for r in cluster_runs if r is not None])
    cluster_runs.to_pickle(save_path / "clustering_metrics.pkl", compression="gzip")
