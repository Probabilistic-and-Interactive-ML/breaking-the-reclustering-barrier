import random
import time
from pathlib import Path

import numpy as np
import torch
import tyro
from shortuuid import uuid
from threadpoolctl import threadpool_limits

import wandb
from config.base_config import Args
from src.datasets.dataset_init import _get_smallest_subset, get_train_eval_test_dataloaders
from src.deep import BRB_DCN, BRB_DEC, BRB_IDEC
from src.deep._torch_utils import set_torch_seed
from src.deep.brb_reclustering import brb_settings_printout, brb_short_printout
from src.deep.evaluation import evaluate_deep_clustering
from src.training._utils import determine_optimizer
from src.training.ae import pretrain_ae
from src.training.ae_init import initialize_autoencoder
from src.training.utils import get_ae_path, get_gpu_with_most_free_memory, get_number_of_clusters, set_cuda_configuration
from src.training.wandb_init import initialize_wandb


@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def train(args: Args) -> None:
    """Training script for training DEC, IDEC, DCN ... ."""

    if args.experiment.result_dir is None:
        result_dir = Path(".").resolve()
    else:
        result_dir = Path(args.experiment.result_dir).resolve()

    # Infer the smallest subset for the dataset if "smallest" is specified
    args.dataset_subset = _get_smallest_subset(args.dataset_name, args.dataset_subset)
    print(f"Dataset subset: {args.dataset_subset}. Dataset name: {args.dataset_name}.")

    # Set some paths
    run_name = (
        f"{args.dataset_name}-{args.dc_algorithm}-{args.model_type}-{args.dc_optimizer.lr}-"
        f"{args.brb.reset_weights}-{args.brb.recluster}-"
        f"{args.brb.reset_momentum}-{args.brb.recalculate_centers}-{uuid()}"
    )
    data_path = result_dir / "data"
    base_path = result_dir / f"experiments/{args.experiment.prefix}/{run_name}"
    # This was not used before. Just passing result_dir is fine, wandb will create a separate "wandb" directory
    wandb_logging_dir = result_dir

    cp_path = base_path / f"seed_{args.seed}"
    cp_path.mkdir(parents=True, exist_ok=True)
    if args.ae_path is None:
        args.ae_path = get_ae_path(args, data_path)
    print("Start: ", base_path)

    # Seeding & reproducibility
    rng = np.random.RandomState(args.seed)
    set_torch_seed(rng)
    # NOTE: This is not deterministic, but it is reproducible
    # There are multiple parts of the code where warnings related to deterministic algorithms are suppressed
    torch.use_deterministic_algorithms(args.experiment.deterministic_torch, warn_only=True)
    # Use bfloat16 for faster computations
    torch.set_float32_matmul_precision("medium")

    # Set correct flags for logging purpose
    if args.use_contrastive_loss:
        # set to True for logging purpose
        args.augmented_pretraining = True

    # check if augmentation invariance is used with contrastive loss
    if args.use_contrastive_loss and not args.augmentation_invariance:
        raise ValueError("If contrastive loss is used augmentation invariance needs to be set to True")

    # Set number of clusters to ground truth number of clusters if n_clusters is not specified
    if args.n_clusters is None:
        args.n_clusters = get_number_of_clusters(args.dataset_name, args.dataset_subset)

    # Set embedding dimension to number of clusters if not specified
    if args.embedding_dim is None:
        args.embedding_dim = args.n_clusters

    # Initialize wandb run
    run = initialize_wandb(args=args, name=run_name, wandb_logging_dir=wandb_logging_dir)

    # Load and preprocess data
    ae_train_dl, _, dc_train_dl, dc_train_wA_dl, test_dl, data, labels, _, test_labels = get_train_eval_test_dataloaders(
        args, data_path
    )
    custom_dataloaders = (dc_train_dl, dc_train_wA_dl, test_dl)

    # Sanity check in case get_number_of_clusters function is not properly maintained
    assert args.n_clusters == len(set(labels.tolist()))

    if isinstance(args.experiment.gpu, tuple | list | str):
        # Randomize starting times to not initially schedule everything on the GPU with the lowest ID
        time.sleep(random.randint(0, 5))
        gpu = get_gpu_with_most_free_memory(args.experiment.gpu)
        device = torch.device(gpu)
        if args.experiment.track_wandb:
            run.log({"System/GPU": gpu})
    else:
        device = set_cuda_configuration(args.experiment.gpu)

    ae = initialize_autoencoder(args=args, data=data, device=device)

    # Print out general BRB settings before training
    brb_short_printout(vars(args.brb))
    brb_settings_printout(vars(args.brb))

    ae = pretrain_ae(
        ae=ae,
        dataloader=ae_train_dl,
        optimizer_args=args.pretrain_optimizer,
        n_epochs=args.pretrain_epochs,
        overwrite_ae=args.overwrite_ae,
        ae_path=args.ae_path,
        save_model=args.save_ae,
        wandb_run=run,
        device=device,
    )

    # Set up the optimizer
    optimizer_class = determine_optimizer(
        optimizer_name=args.dc_optimizer.optimizer, weight_decay=args.dc_optimizer.weight_decay
    )

    if args.clustering_epochs > 0:
        if args.dc_algorithm == "dec":
            cluster_algo = BRB_DEC(
                n_clusters=args.n_clusters,
                clustering_epochs=args.clustering_epochs,
                autoencoder=ae,
                optimizer_class=optimizer_class,
                clustering_optimizer_params=args.dc_optimizer,
                augmentation_invariance=args.augmentation_invariance,
                custom_dataloaders=custom_dataloaders,
                reset_args=args.brb,
                device=device,
                wandb_run=run,
                log_interval=args.experiment.cluster_log_interval,
                num_workers=args.experiment.num_workers_dataloader,
                track_silhouette=args.experiment.track_silhouette,
                track_purity=args.experiment.track_purity,
                track_voronoi=args.experiment.track_voronoi,
                track_uncertainty_plot=args.experiment.track_uncertainty_plot,
            )
        elif args.dc_algorithm == "idec":
            cluster_algo = BRB_IDEC(
                n_clusters=args.n_clusters,
                clustering_epochs=args.clustering_epochs,
                cluster_loss_weight=args.cluster_loss_weight,
                data_reg_loss_weight=args.data_reg_loss_weight,
                augmentation_invariance=args.augmentation_invariance,
                custom_dataloaders=custom_dataloaders,
                autoencoder=ae,
                optimizer_class=optimizer_class,
                clustering_optimizer_params=args.dc_optimizer,
                reset_args=args.brb,
                device=device,
                wandb_run=run,
                log_interval=args.experiment.cluster_log_interval,
                num_workers=args.experiment.num_workers_dataloader,
                track_silhouette=args.experiment.track_silhouette,
                track_purity=args.experiment.track_purity,
                track_voronoi=args.experiment.track_voronoi,
                track_uncertainty_plot=args.experiment.track_uncertainty_plot,
            )
        elif args.dc_algorithm == "dcn":
            cluster_algo = BRB_DCN(
                n_clusters=args.n_clusters,
                clustering_epochs=args.clustering_epochs,
                cluster_loss_weight=args.cluster_loss_weight,
                # NOTE: reconstruction_loss_weight is weighing both the reconstruction loss and contrastive loss
                representation_loss_weight=args.data_reg_loss_weight,
                augmentation_invariance=args.augmentation_invariance,
                custom_dataloaders=custom_dataloaders,
                autoencoder=ae,
                optimizer_class=optimizer_class,
                clustering_optimizer_params=args.dc_optimizer,
                reset_args=args.brb,
                device=device,
                wandb_run=run,
                log_interval=args.experiment.cluster_log_interval,
                num_workers=args.experiment.num_workers_dataloader,
                track_silhouette=args.experiment.track_silhouette,
                track_purity=args.experiment.track_purity,
                track_voronoi=args.experiment.track_voronoi,
                track_uncertainty_plot=args.experiment.track_uncertainty_plot,
            )
        else:
            raise ValueError(f"Invalid clustering algorithm {args.dc_algorithm} specified.")

        if args.dataset_subset == "train":
            # This is the case for contrastive methods that usually use the train and test split
            labels_to_use = test_labels.tolist()
        else:
            # This is the case for autoencoder based methods, where evaluation is usually done on the same data
            labels_to_use = labels.tolist()
        # Fit the algorithm
        cluster_algo.fit(X=None, y=labels_to_use)
        cluster_algo.autoencoder.to("cpu")
        # optimizers with lambda functions cannot be pickled, thus they are set to None:
        cluster_algo.optimizer_class = None

        # Evaluate final performance on test set
        metrics, _ = evaluate_deep_clustering(
            cluster_centers=cluster_algo.dc_cluster_centers_,
            model=cluster_algo.autoencoder.to(device),
            dataloader=test_dl,
            labels=test_labels,
            old_labels=None,
            loss_fn=torch.nn.MSELoss(),
            metrics_dict=None,
            return_labels=False,
            track_silhouette=True,
            track_purity=True,
            device=device,
            track_voronoi=args.experiment.track_voronoi,
            track_uncertainty_plot=args.experiment.track_uncertainty_plot,
        )

        # Log final scores on test set
        if run is not None:
            # Skip Cluster_Change metrics as we are not interested in those
            metric_dict = {f"Clustering test metrics/{k}": v[-1] for k, v in metrics.items() if "Change" not in k}
            metric_dict["Clustering epoch"] = args.clustering_epochs
            run.log(metric_dict)

    else:
        print("Deep Clustering is skipped")

    if args.save_clustering_model:
        brb = "baseline"
        if args.reset_weights:
            brb = "brb"

        torch.save(
            {
                "sd": cluster_algo.autoencoder.state_dict(),
                "kmeans_centers": cluster_algo.cluster_centers_,
                "kmeans_labels": cluster_algo.labels_,
                "dc_centers": cluster_algo.dc_cluster_centers_,
                "dc_labels": cluster_algo.dc_labels_,
            },
            f"{data_path}/clustering_models/{args.dc_algorithm}_{args.dataset_name}_{brb}_{args.seed}.pth",
        )
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
