from pathlib import Path

import torch

from config.optimizer_config import PretrainOptimizerArgs
from src.deep._optimizer_utils import LinearWarmupCosineAnnealingLRScheduler
from src.deep.autoencoders import FeedforwardAutoencoder
from src.training._utils import determine_optimizer
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


def pretrain_ae(
    ae: FeedforwardAutoencoder,
    dataloader: torch.utils.data.DataLoader | None,
    optimizer_args: PretrainOptimizerArgs,
    n_epochs: int,
    overwrite_ae: bool,
    ae_path: str | Path,
    save_model: bool,
    device: torch.device,
    wandb_run: Run | RunDisabled | None = None,
) -> FeedforwardAutoencoder:
    """Pretrain an autoencoder on the given data set."""

    print("ae_path: ", ae_path)
    # train model if it does not exist yet or we do not want to load it, i.e., retrain it
    ae_path = Path(ae_path)

    if n_epochs == 0:
        ae.fitted = True
        return ae
    elif (not ae_path.resolve().exists() or overwrite_ae) and n_epochs > 0:
        scheduler = None
        scheduler_params = None
        if optimizer_args.schedule_lr:
            scheduler = LinearWarmupCosineAnnealingLRScheduler
            scheduler_params = {
                "warmup_epochs": int(n_epochs * 0.2),
                "total_epochs": n_epochs,
                "T_0": n_epochs,
                "T_mult": 1,
                "start_factor": 0.1,
                "end_factor": 1.0,
            }
        optimizer_class = determine_optimizer(
            optimizer_name=optimizer_args.optimizer, weight_decay=optimizer_args.weight_decay
        )
        ae.fit(
            n_epochs=n_epochs,
            optimizer_params=optimizer_args,
            dataloader=dataloader,
            device=device,
            print_step=10,
            optimizer_class=optimizer_class,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            wandb_run=wandb_run,
        )
        ae.eval()

        if save_model:
            # Check if directory exists and create it if necessary
            ae_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ae.cpu().state_dict(), ae_path)
    else:
        # Load AE
        sd = torch.load(ae_path, map_location=device)
        ae.load_state_dict(sd)
        ae.fitted = True
        ae.eval()
        print(f"Pretrained Autoencoder loaded from {ae_path}.")
    ae.to(device)

    return ae
