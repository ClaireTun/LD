"""Invariant Feature Adapter for DriveMem."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class InvariantFeatureAdapter(nn.Module):
    """Produces invariant and spurious query tokens from side-branch features."""

    def __init__(self, input_dim: int, ifa_dim: int, hidden_dim: int, pooling: str = "mean", dropout: float = 0.0) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported IFA pooling={pooling}")
        self.pooling = pooling
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.inv_proj = nn.Linear(hidden_dim, ifa_dim)
        self.spu_proj = nn.Linear(hidden_dim, ifa_dim)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if self.pooling == "mean":
                return x.mean(dim=1)
            return x[:, 0]
        if x.ndim == 2:
            return x
        raise ValueError(f"IFA expects [B, N, D] or [B, D], got shape={tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self._pool(x)
        first_weight = self.backbone[1].weight
        pooled = pooled.to(device=first_weight.device, dtype=first_weight.dtype)
        h = self.backbone(pooled)
        z_inv = self.inv_proj(h)
        z_spu = self.spu_proj(h)
        if z_inv.ndim != 2 or z_spu.shape != z_inv.shape:
            raise RuntimeError("IFA produced invalid invariant/spurious shapes")
        return z_inv, z_spu


def orthogonality_loss(z_inv: torch.Tensor, z_spu: torch.Tensor) -> torch.Tensor:
    if z_inv.shape != z_spu.shape or z_inv.ndim != 2:
        raise ValueError(f"Orthogonality loss expects matching [B, D], got {tuple(z_inv.shape)} and {tuple(z_spu.shape)}")
    inv = F.normalize(z_inv.float(), dim=-1)
    spu = F.normalize(z_spu.float(), dim=-1)
    return (inv * spu).sum(dim=-1).pow(2).mean()


def dsu_consistency_loss(z_inv_dsu: torch.Tensor, z_inv_clean: torch.Tensor) -> torch.Tensor:
    if z_inv_dsu.shape != z_inv_clean.shape:
        raise ValueError("DSU consistency expects matching invariant query shapes")
    dsu = F.normalize(z_inv_dsu.float(), dim=-1)
    clean = F.normalize(z_inv_clean.detach().float(), dim=-1)
    return (dsu - clean).pow(2).sum(dim=-1).mean()
