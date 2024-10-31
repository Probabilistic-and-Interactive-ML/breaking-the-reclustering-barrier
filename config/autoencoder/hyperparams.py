from config.runner_config import BRBArgs, DCOptimizerArgs, ExperimentArgs, PretrainOptimizerArgs, RunnerArgs

config = {
    ######################################################################
    # Base hyperparameter sensitivity analysis for BRB
    ######################################################################
    "brb_hyperparameters_base": RunnerArgs(
        command="python train.py",
        workers=1,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - Base Hyperparameter Analysis"',
            result_dir=None,
            prefix="hyperparameters_base",
            tag=("hyperparameters_base"),
            gpu=0,
            track_wandb=True,
            debug=False,
            wandb_entity=None,
        ),
        seed=(400, 148, 820, 214, 40), 
        dataset_subset="all",
        # use reduced set of datasets
        dataset_name="mnist",
        model_type="feedforward_large",
        activation_fn="relu",
        pretrain_optimizer=PretrainOptimizerArgs(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        embedding_dim=None,  # 5, 50, 100
        dc_algorithm=("idec", "dcn", "dec"),
        dc_optimizer=DCOptimizerArgs(
            optimizer="adam",
            lr=(5e-3, 1e-3, 5e-4, 1e-4),
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="none",
        ),
        pretrain_epochs=250,
        clustering_epochs=400,
        augmented_pretraining=False,
        augmentation_invariance=True,
        # brb parameters
        brb=BRBArgs(
            subsample_size=10000,
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            reset_momentum=True,
            reset_interpolation_factor=(0.6, 0.7, 0.8, 0.9),
            reset_interval=(5, 10, 20, 40),
            reset_embedding=False,
        ),
    ),
}
