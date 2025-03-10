from config.runner_config import RunnerArgs
from config.base_config import (
    ExperimentArgs,
    PretrainOptimizerArgs, 
    DCOptimizerArgs, 
    BRBArgs)

config = {
    "dec_cifar100_20_brb": RunnerArgs(
        command="python train.py",
        experiment=ExperimentArgs(),
        seed=(
            214, 148, 820, 400, 40, 805, 111, 491, 215, 513
        ), 
        model_type="feedforward_medium",
        augment_both_views=True,
        use_contrastive_loss=True,
        augmented_pretraining=True,
        augmentation_invariance=True,
        batch_norm=True,
        activation_fn="relu",
        embedding_dim=128,
        dc_algorithm="dec",
        convnet="resnet18",
        dataset_name="cifar100_20",
        dataset_subset="train",
        pretrain_epochs=1000,
        clustering_epochs=1000,
        projector_depth=1,
        projector_layer_size=2048,
        cluster_loss_weight=1.0,
        data_reg_loss_weight=1.0,
        softmax_temperature_tau=0.5,
        crop_size=32,
        batch_size=512,
        n_clusters=None,
        pretrain_optimizer=PretrainOptimizerArgs(
            lr=3e-3,
            weight_decay=1e-4,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        dc_optimizer=DCOptimizerArgs(
            lr=1e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="linear",
            scheduler_warmup_epochs=10,
            scheduler_start_factor=0.01,
            scheduler_end_factor=1.0,
            reset_lr_scheduler=False,
        ),
        brb=BRBArgs(
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            subsample_size=10000,
            reset_interpolation_factor=0.7,
            reset_interpolation_factor_step=0.2,
            reset_convlayers="4",
            reset_interval=10,
            reset_embedding=False,
            reset_momentum=True,
        ),
    ),
    "idec_cifar100_20_brb": RunnerArgs(
        command="python train.py",
        experiment=ExperimentArgs(),
        seed=(
            214, 148, 820, 400, 40, 805, 111, 491, 215, 513
        ), 
        model_type="feedforward_medium",
        augment_both_views=True,
        use_contrastive_loss=True,
        augmented_pretraining=True,
        augmentation_invariance=True,
        batch_norm=True,
        activation_fn="relu",
        embedding_dim=128,
        dc_algorithm="idec",
        convnet="resnet18",
        dataset_name="cifar100_20",
        dataset_subset="train",
        pretrain_epochs=1000,
        clustering_epochs=1000,
        projector_depth=1,
        projector_layer_size=2048,
        cluster_loss_weight=1.0,
        data_reg_loss_weight=1.0,
        softmax_temperature_tau=0.5,
        crop_size=32,
        batch_size=512,
        n_clusters=None,
        pretrain_optimizer=PretrainOptimizerArgs(
            lr=3e-3,
            weight_decay=1e-4,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        dc_optimizer=DCOptimizerArgs(
            lr=3e-3,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="linear",
            scheduler_warmup_epochs=10,
            scheduler_start_factor=0.01,
            scheduler_end_factor=1.0,
            reset_lr_scheduler=False,
        ),
        brb=BRBArgs(
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            subsample_size=10000,
            reset_interpolation_factor=0.7,
            reset_interpolation_factor_step=0.2,
            reset_convlayers="4",
            reset_interval=10,
            reset_embedding=False,
            reset_momentum=True,
        ),
    ),
        "dcn_cifar100_20_brb": RunnerArgs(
        command="python train.py",
        experiment=ExperimentArgs(),
        seed=(
            214, 148, 820, 400, 40, 805, 111, 491, 215, 513
        ), 
        model_type="feedforward_medium",
        augment_both_views=True,
        use_contrastive_loss=True,
        augmented_pretraining=True,
        augmentation_invariance=True,
        batch_norm=True,
        activation_fn="relu",
        embedding_dim=128,
        dc_algorithm="dcn",
        convnet="resnet18",
        dataset_name="cifar100_20",
        dataset_subset="train",
        pretrain_epochs=1000,
        clustering_epochs=1000,
        projector_depth=1,
        projector_layer_size=2048,
        cluster_loss_weight=1e-4,
        data_reg_loss_weight=1.0,
        softmax_temperature_tau=0.5,
        crop_size=32,
        batch_size=512,
        n_clusters=None,
        pretrain_optimizer=PretrainOptimizerArgs(
            lr=3e-3,
            weight_decay=1e-4,
            projector_weight_decay=0.0,
            schedule_lr=False,
        ),
        dc_optimizer=DCOptimizerArgs(
            lr=1e-2,
            weight_decay=0.0,
            projector_weight_decay=0.0,
            scheduler_name="linear",
            scheduler_warmup_epochs=10,
            scheduler_start_factor=0.01,
            scheduler_end_factor=1.0,
            reset_lr_scheduler=False,
        ),
        brb=BRBArgs(
            reset_weights=True,
            recluster=True,
            recalculate_centers=False,
            subsample_size=10000,
            reset_interpolation_factor=0.7,
            reset_interpolation_factor_step=0.2,
            reset_convlayers="4",
            reset_interval=10,
            reset_embedding=False,
            reset_momentum=True,
        ),
    )
}