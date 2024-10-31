from config.runner_config import BRBArgs, DCOptimizerArgs, ExperimentArgs, PretrainOptimizerArgs, RunnerArgs

config = {
    ######################################################################
    # From scratch training (pretraining_epochs=0) with reclustering variations
    ######################################################################
    "no_pretraining_reduced_brb": RunnerArgs(
        command="python train.py",
        workers=1,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - No Pretraining"',
            result_dir=None,
            prefix="no_pretraining",
            tag=("no_pretraining"),
            gpu=0,
            track_wandb=True,
            debug=False,
            wandb_entity=None,
        ),
        seed=(400, 148, 820, 214, 40, 805, 111, 491, 215, 513),
        dataset_subset="all",
        dataset_name=("mnist", "optdigits", "usps", "fmnist", "kmnist", "gtsrb"),
        model_type="feedforward_large",
        activation_fn="relu",
        embedding_dim=None,  # 5, 50, 100
        dc_algorithm=("idec", "dcn", "dec"),
        pretrain_optimizer=PretrainOptimizerArgs(),
        dc_optimizer=DCOptimizerArgs(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="none",
        ),
        pretrain_epochs=0,
        # Increase clustering epochs
        clustering_epochs=400,
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
