from config.runner_config import BRBArgs, DCOptimizerArgs, ExperimentArgs, PretrainOptimizerArgs, RunnerArgs

config = {
    "no_pretraining": RunnerArgs(
        command="python train.py",
        workers=1,
        save_ae=True,
        overwrite_ae=False,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - Comparison No Pretraining - REPRO"',
            wandb_entity="kevin-sidak-team",
            result_dir=None,
            prefix="no_pretraining",
            tag=("no_pretraining"),
            gpu=0,
            num_workers_dataloader=1,
            track_wandb=True,
            debug=False,
        ),
        seed=(400, 148, 820, 214, 40, 805, 111, 491, 215, 513),
        dataset_subset="all",
        dataset_name=(
            "optdigits",
            "usps",
            "gtsrb",
            "fmnist",
            "mnist",
            "kmnist",
        ),  
        model_type="feedforward_large",
        activation_fn="relu",
        embedding_dim=None,
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
        # BRB parameters
        # Configurations for "baselines", "reset_only" and "recluster_only" shown below
        brb=BRBArgs(
            subsample_size=10000,
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            reset_momentum=True,
            reset_interpolation_factor=0.8,
            reset_interval=20,
            reset_embedding=False,
        ),
        # # reset only
        # brb=BRBArgs(
        #     subsample_size=10000,
        #     reset_weights=True,
        #     recluster=False,
        #     recalculate_centers=False,
        #     reset_momentum=False,
        #     reset_interpolation_factor=0.8,
        #     reset_interval=20,
        #     reset_embedding=False,
        # ),
        # # recluster only
        # brb=BRBArgs(
        #     subsample_size=10000,
        #     reset_weights=False,
        #     recluster=True,
        #     recalculate_centers=False,
        #     reset_momentum=False,
        #     reset_interpolation_factor=0.8,
        #     reset_interval=20,
        #     reset_embedding=False,
        # ),
        # # Baselines without intervention
        # brb=BRBArgs(
        #     subsample_size=10000,
        #     reset_weights=False,
        #     recluster=False,
        #     recalculate_centers=False,
        #     reset_momentum=False,
        #     reset_embedding=False,
        # ),
    )
}
