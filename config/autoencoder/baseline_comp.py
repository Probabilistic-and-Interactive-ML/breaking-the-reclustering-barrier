from config.runner_config import BRBArgs, DCOptimizerArgs, ExperimentArgs, PretrainOptimizerArgs, RunnerArgs

config = {
    "baseline_comparison": RunnerArgs(
        command="python train.py",
        workers=1,
        save_ae=True,
        overwrite_ae=False,
        experiment=ExperimentArgs(
            wandb_project='"Clustering BRB - Comparison - REPRO"',
            wandb_entity="kevin-sidak-team",
            result_dir=None,
            prefix="baseline_comparison",
            tag=("baseline_comparison"),
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
        # Configurations for "baselines", "reset_only" and "recluster_only" shown below
        # # BRB parameters
        # brb=BRBArgs(
        #     subsample_size=10000,
        #     reset_weights=True,
        #     recluster=True,
        #     recalculate_centers=False,
        #     reset_momentum=True,
        #     reset_interpolation_factor=0.8,
        #     reset_interval=20,
        #     reset_embedding=False,
        # ),
        # # reset only
        brb=BRBArgs(
            subsample_size=10000,
            reset_weights=True,
            recluster=False,
            recalculate_centers=False,
            reset_momentum=False,
            reset_interpolation_factor=0.8,
            reset_interval=20,
            reset_embedding=False,
        ),
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
        # Baselines without intervention
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
