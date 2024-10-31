import torch
from torch.optim.lr_scheduler import _LRScheduler

from config.optimizer_config import DCOptimizerArgs, PretrainOptimizerArgs


def param_groups_weight_decay(
    model: torch.nn.Module,
    weight_decay=0.0,
    no_weight_decay_list=(),
    skip_list=(),
):
    """
    Splits param_groups into one group using weight_decay and another group without weight decay.
    Returns a list of param groups to be passed to the optimizer
    """
    # taken from https://github.com/huggingface/pytorch-image-models/blob/6e6f3686a7e06bcba37bbd3b7c755f04a516a1e7/timm/optim/optim_factory.py#L42
    no_weight_decay_list = set(no_weight_decay_list)
    skip_list = set(skip_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # skip all frozen parameters
            continue

        skip_param = _is_substring_in_list(name, skip_list)
        if not skip_param:
            # skip all mentioned params
            skip_wd = _is_substring_in_list(name, no_weight_decay_list)
            if param.ndim <= 1 or name.endswith(".bias") or skip_wd:
                no_decay.append(param)
            else:
                decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def _is_substring_in_list(input: str, l: list):
    """Checks if input string is substring of strings in list. Case Insensitive."""
    for value in l:
        if value.lower() in input.lower():
            return True
    return False


# Implementation inspired by this post: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
def setup_optimizer(
    model: torch.nn.Module,
    dc_module: torch.nn.Module | None,
    optimizer_class: torch.optim.Optimizer,
    optimizer_params: PretrainOptimizerArgs | DCOptimizerArgs,
    freeze_convlayers: bool,
):
    param_dicts = []
    # freeze convlayers
    if freeze_convlayers:
        for param_name, param_group in model.named_parameters():
            if "conv_" in param_name:
                param_group.requires_grad = False

    # handle general weight decay
    param_dicts += param_groups_weight_decay(model, optimizer_params.weight_decay, skip_list=["projector"])
    # handle weight decay for projector
    param_dicts += param_groups_weight_decay(model.projector, optimizer_params.projector_weight_decay)
    # IMPORTANT: Do not change the order of how the parameters are added (dec_module must be last),
    # due to behaviour of adam_reset_centers_momentum function, which assumes centers are last in the optimizer
    # cluster centers are added last and do not use weight decay
    if dc_module is not None:
        param_dicts += [{"params": dc_module.parameters()}]
    # pass all parameters to optimizer
    optimizer = optimizer_class(param_dicts, lr=optimizer_params.lr)
    return optimizer


def init_lr_scheduler(
    optimizer: torch.optim.Optimizer, lr_scheduler_args: DCOptimizerArgs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Initialize learning rate scheduler for the deep clustering phase."""

    correct_lr_scheduler_args = {}
    if lr_scheduler_args.scheduler_name == "none":
        lr_scheduler = None
    elif lr_scheduler_args.scheduler_name == "step":
        correct_lr_scheduler_args = {
            "step_size": lr_scheduler_args.scheduler_step_size,
            "gamma": lr_scheduler_args.scheduler_gamma,
        }
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **correct_lr_scheduler_args)
    elif lr_scheduler_args.scheduler_name == "cosine":
        correct_lr_scheduler_args = {"T_0": lr_scheduler_args.scheduler_T_0, "T_mult": lr_scheduler_args.scheduler_T_mult}
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **correct_lr_scheduler_args)
    elif lr_scheduler_args.scheduler_name == "linear_warmup_cosine":
        correct_lr_scheduler_args = {
            "warmup_epochs": lr_scheduler_args.scheduler_warmup_epochs,
            "total_epochs": lr_scheduler_args.scheduler_total_epochs,
            "T_0": lr_scheduler_args.scheduler_T_0,
            "T_mult": lr_scheduler_args.scheduler_T_mult,
            "start_factor": lr_scheduler_args.scheduler_start_factor,
            "end_factor": lr_scheduler_args.scheduler_end_factor,
        }
        lr_scheduler = LinearWarmupCosineAnnealingLRScheduler(optimizer, **correct_lr_scheduler_args)
    elif lr_scheduler_args.scheduler_name == "linear":
        correct_lr_scheduler_args = {
            "start_factor": lr_scheduler_args.scheduler_start_factor,
            "end_factor": lr_scheduler_args.scheduler_end_factor,
            "total_iters": lr_scheduler_args.scheduler_total_epochs,
        }
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **correct_lr_scheduler_args)
    else:
        raise ValueError(f"Invalid lr_scheduler {lr_scheduler_args.scheduler_name} specified.")
    return lr_scheduler


class LinearWarmupCosineAnnealingLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        T_0,
        end_factor=1.0,
        T_mult=1,
        last_epoch=-1,
        start_factor=0.1,
        verbose=False,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.last_epoch = 0
        self._last_lr = 0
        self.epoch_counter = 0
        self.linear_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.start_factor,
            end_factor=self.end_factor,
            total_iters=warmup_epochs,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        self.cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, last_epoch=last_epoch - warmup_epochs, verbose=verbose
        )
        super(LinearWarmupCosineAnnealingLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch
        self.last_epoch = epoch
        if epoch < self.warmup_epochs:
            self.linear_lr.step(epoch)
            self._last_lr = self.get_last_lr()
        else:
            if epoch == self.warmup_epochs:
                self.cosine_annealing.T_cur = epoch - self.warmup_epochs
            self.cosine_annealing.step(epoch - self.warmup_epochs)
            self._last_lr = self.get_last_lr()
        self.last_epoch += 1

    def get_last_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.linear_lr.get_last_lr()
        else:
            return self.cosine_annealing.get_last_lr()
