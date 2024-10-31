import warnings

import torch


# Adapted from https://github.com/DanielTrosten/DeepMVC/blob/main/src/models/comvc/comvc_loss.py
class SimclrLoss(torch.nn.Module):
    """Implements SimCLR loss for 2 views"""

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.large_num = 1e9

    def _contrastive_loss_two_views(self, h1, h2):
        """
        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py
        """
        device = h1.device
        n = h1.size(0)
        labels = torch.arange(0, n, device=device, dtype=torch.long)
        masks = torch.eye(n, device=device)

        # Suppress the warnings for deterministic torch that clutter the output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            logits_aa = ((h1 @ h1.t()) / self.tau) - masks * self.large_num
            logits_bb = ((h2 @ h2.t()) / self.tau) - masks * self.large_num

            logits_ab = (h1 @ h2.t()) / self.tau
            logits_ba = (h2 @ h1.t()) / self.tau

        loss_a = torch.nn.functional.cross_entropy(torch.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat((logits_ba, logits_bb), dim=1), labels)

        loss = loss_a + loss_b
        return loss

    def forward(self, projections: list) -> torch.float:
        z = torch.stack(projections, dim=0)
        z = torch.nn.functional.normalize(z, dim=-1, p=2)
        loss = self._contrastive_loss_two_views(z[0], z[1])
        return loss
