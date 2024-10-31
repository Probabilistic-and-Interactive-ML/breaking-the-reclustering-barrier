import torch


def determine_optimizer(optimizer_name: str, weight_decay: float = None) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        # SGD expects weight_decay of type float and not None
        optimizer_class = torch.optim.SGD
    elif weight_decay is not None and weight_decay > 0 and optimizer_name == "adam":
        optimizer_class = torch.optim.AdamW
    elif optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError(f"Invalid optimizer {optimizer_name} specified.")
    return optimizer_class
