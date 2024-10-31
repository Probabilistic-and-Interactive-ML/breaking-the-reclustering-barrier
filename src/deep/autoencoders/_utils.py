import torch


def split_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # assume views of same size
    split_idx = x.shape[0] // 2
    view1 = x[:split_idx]
    view2 = x[split_idx:]
    assert view1.shape == view2.shape
    return view1, view2
