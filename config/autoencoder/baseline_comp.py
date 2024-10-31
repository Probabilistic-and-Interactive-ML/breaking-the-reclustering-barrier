from config.runner_config import BRBArgs, DCOptimizerArgs, ExperimentArgs, PretrainOptimizerArgs, RunnerArgs

config = {
    "baseline_comparison": RunnerArgs(
        command="python train.py",
        workers=1,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - Baseline Comparison"',
            result_dir=None,
            prefix="baseline_comparison",
            tag=("baseline_comparison"),
            gpu=0,
            track_wandb=True,
            debug=False,
            wandb_entity=None,
        ),
        seed=(400, 148, 820, 214, 40, 805, 111, 491, 215, 513),
        dataset_subset="all",
        dataset_name=(
            "mnist",
            "optdigits",
            "cifar10",
            "gtsrb",
            "fmnist",
            "kmnist",
            "usps",
        ),  
        model_type="feedforward_large",
        activation_fn="relu",
        pretrain_optimizer=PretrainOptimizerArgs(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        dc_optimizer=DCOptimizerArgs(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="none",
        ),
        embedding_dim=None,
        dc_algorithm=("idec", "dcn", "dec"),
        pretrain_epochs=250,
        clustering_epochs=400,
        augmented_pretraining=False,
        augmentation_invariance=True,
        # brb parameters
        brb=BRBArgs(
            subsample_size=10000,
            # Also do for "reset_only", "recluster_only"
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            reset_momentum=True,
            reset_interpolation_factor=0.8,
            reset_interval=20,
            reset_embedding=False,
        ),
    )
}
