"""Losses that preserve the SPM log-probability objective."""

import torch
from torch import nn


class LabelSmoothingNLLLoss(nn.Module):
    """NLL with a uniform target distribution, for SPM log probabilities."""

    def __init__(self, smoothing: float):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        uniform = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


def mixed_nll(
    criterion: nn.Module,
    log_probs: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    return coefficient * criterion(log_probs, target_a) + (1 - coefficient) * criterion(
        log_probs, target_b
    )
