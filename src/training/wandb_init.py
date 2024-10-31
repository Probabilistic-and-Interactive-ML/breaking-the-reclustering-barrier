from copy import deepcopy
from pathlib import Path

import wandb
from config.base_config import Args
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


def initialize_wandb(args: Args, name: str, wandb_logging_dir: str | Path) -> Run | RunDisabled:
    """Initialize wandb and return the run object and the args object.

    When the run is a wandb sweep, the args are instantiated here.
    Sets up the different x-axis for the wandb plots.
    """

    # Set up the tags to be used for the wandb run
    if isinstance(args.experiment.tag, str):
        tag = [args.experiment.tag]
    elif isinstance(args.experiment.tag, tuple):
        # Tyro uses a tuple for tags, wandb only allows lists
        tag = list(args.experiment.tag)
    else:
        assert args.experiment.tag is None
        tag = None

    # Convert the config to a dictionary for wandb
    # deepcopy is necessary because vars is a reference to the object
    dict_args = vars(deepcopy(args))
    dict_args["experiment"] = vars(dict_args["experiment"])
    dict_args["pretrain_optimizer"] = vars(dict_args["pretrain_optimizer"])
    dict_args["dc_optimizer"] = vars(dict_args["dc_optimizer"])
    dict_args["brb"] = vars(dict_args["brb"])

    run = wandb.init(
        dir=wandb_logging_dir,
        project=args.experiment.wandb_project,
        entity=args.experiment.wandb_entity,
        name=name,
        tags=tag,
        config=dict_args,
        mode="online" if args.experiment.track_wandb else "disabled",
    )

    # AE epochs and clustering epochs
    wandb.define_metric("AE train epoch")
    wandb.define_metric("AE plots/*", step_metric="AE train epoch")
    wandb.define_metric("Clustering epoch")
    wandb.define_metric("Clustering metrics/*", step_metric="Clustering epoch")
    wandb.define_metric("Clustering train/*", step_metric="Clustering epoch")
    wandb.define_metric("Clustering test metrics/*", step_metric="Clustering epoch")

    return run
