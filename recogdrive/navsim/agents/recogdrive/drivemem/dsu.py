"""Distribution uncertainty modules for DriveMem side branches."""
from __future__ import annotations

import torch
from torch import nn


class TokenDSU(nn.Module):
    """Token-wise DSU for ViT/VLM token features shaped [B, N, D].

    The module has no learnable parameters and is active only in training.
    It must be used only on DriveMem side-branch features, never on the clean
    visual/VLM/planner main path.
    """

    def __init__(self, p: float = 0.05, eps: float = 1e-6, factor: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TokenDSU expects [B, N, D], got shape={tuple(x.shape)}")
        if (not self.training) or self.p <= 0.0 or self.factor <= 0.0:
            return x
        if torch.rand((), device=x.device) >= self.p:
            return x

        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)

        # Batch-level uncertainty of token statistics. Keep a minimum epsilon so
        # batch size 1 remains stable and simply yields near-clean perturbations.
        mean_unc = mean.var(dim=0, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        std_unc = std.var(dim=0, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)

        beta = mean + self.factor * mean_unc * torch.randn_like(mean)
        gamma = (std + self.factor * std_unc * torch.randn_like(std)).clamp_min(self.eps)
        x_norm = (x - mean) / std
        return x_norm * gamma + beta


class DSU2D(nn.Module):
    """2D DSU for CNN feature maps shaped [B, C, H, W]."""

    def __init__(self, p: float = 0.05, eps: float = 1e-6, factor: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DSU2D expects [B, C, H, W], got shape={tuple(x.shape)}")
        if (not self.training) or self.p <= 0.0 or self.factor <= 0.0:
            return x
        if torch.rand((), device=x.device) >= self.p:
            return x

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
        mean_unc = mean.var(dim=0, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        std_unc = std.var(dim=0, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        beta = mean + self.factor * mean_unc * torch.randn_like(mean)
        gamma = (std + self.factor * std_unc * torch.randn_like(std)).clamp_min(self.eps)
        x_norm = (x - mean) / std
        return x_norm * gamma + beta
