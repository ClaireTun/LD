"""Single-space learnable prototype memory bank for DriveMem."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PrototypeMemoryBank(nn.Module):
    def __init__(
        self,
        num_prototypes: int = 128,
        memory_dim: int = 256,
        top_k: int = 8,
        temperature: float = 0.07,
        normalize_query: bool = True,
        normalize_memory: bool = True,
        eps: float = 1e-6,
        memory_weight_mode: str = "soft",
        typical_critical_gamma: float = 0.8,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0 or memory_dim <= 0:
            raise ValueError("num_prototypes and memory_dim must be positive")
        self.num_prototypes = int(num_prototypes)
        self.memory_dim = int(memory_dim)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.normalize_query = bool(normalize_query)
        self.normalize_memory = bool(normalize_memory)
        self.eps = float(eps)
        self.memory_weight_mode = memory_weight_mode
        self.typical_critical_gamma = float(typical_critical_gamma)
        self.memory = nn.Parameter(torch.randn(self.num_prototypes, self.memory_dim) * 0.02)
        self.register_buffer("usage_frequency", torch.zeros(self.num_prototypes), persistent=False)
        self.register_buffer("avg_typicality", torch.zeros(()), persistent=False)
        self.register_buffer("avg_criticality", torch.zeros(()), persistent=False)
        self.register_buffer("avg_entropy", torch.zeros(()), persistent=False)

    def forward(self, z_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if z_query.ndim != 2 or z_query.shape[-1] != self.memory_dim:
            raise ValueError(f"PMB expects z_query [B, {self.memory_dim}], got {tuple(z_query.shape)}")
        z_query = z_query.to(device=self.memory.device, dtype=self.memory.dtype)
        query = F.normalize(z_query, dim=-1) if self.normalize_query else z_query
        memory_for_sim = F.normalize(self.memory, dim=-1) if self.normalize_memory else self.memory
        sim = query @ memory_for_sim.t()
        k = self.top_k if self.top_k > 0 else self.num_prototypes
        if k < self.num_prototypes:
            vals, idx = sim.topk(k, dim=-1)
            masked = torch.full_like(sim, float("-inf"))
            sim = masked.scatter(dim=-1, index=idx, src=vals)
        else:
            idx = sim.topk(min(k, self.num_prototypes), dim=-1).indices
        assignment = torch.softmax(sim / max(self.temperature, self.eps), dim=-1)
        z_mem_inv = assignment @ self.memory
        entropy = -(assignment * (assignment + self.eps).log()).sum(dim=-1)
        typicality = 1.0 - entropy / math.log(self.num_prototypes)
        with torch.no_grad():
            winners = assignment.argmax(dim=-1)
            self.usage_frequency.mul_(0.99)
            self.usage_frequency.scatter_add_(0, winners, torch.ones_like(winners, dtype=self.usage_frequency.dtype))
            self.avg_typicality.mul_(0.99).add_(0.01 * typicality.detach().mean())
            self.avg_entropy.mul_(0.99).add_(0.01 * entropy.detach().mean())
        logs = {"assignment_entropy": entropy.detach(), "typicality": typicality.detach(), "topk_indices": idx.detach()}
        return z_mem_inv, assignment, logs

    def loss(self, z_query: torch.Tensor, z_mem_inv: torch.Tensor, assignment: torch.Tensor, per_sample_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        entropy = -(assignment * (assignment + self.eps).log()).sum(dim=-1)
        typicality = 1.0 - entropy / math.log(self.num_prototypes)
        if per_sample_loss is not None:
            loss_i = per_sample_loss.detach().float().view(-1)
            if loss_i.numel() != z_query.shape[0]:
                criticality = torch.zeros_like(typicality)
            else:
                criticality = torch.relu((loss_i - loss_i.mean()) / (loss_i.std(unbiased=False) + self.eps)).to(typicality.device)
        else:
            criticality = torch.zeros_like(typicality)
        with torch.no_grad():
            self.avg_criticality.mul_(0.99).add_(0.01 * criticality.detach().mean())
        if self.memory_weight_mode == "hard":
            t_rank = typicality.argsort().argsort().float() / max(typicality.numel() - 1, 1)
            h_rank = criticality.argsort().argsort().float() / max(criticality.numel() - 1, 1)
            alpha = (torch.maximum(t_rank, h_rank) >= self.typical_critical_gamma).float().detach()
        elif self.memory_weight_mode == "soft":
            raw = typicality + criticality
            alpha = ((raw - raw.min()) / (raw.max() - raw.min() + self.eps)).detach()
            if alpha.sum() <= self.eps:
                alpha = torch.ones_like(alpha)
        else:
            raise ValueError(f"Unsupported memory_weight_mode={self.memory_weight_mode}")
        recon = (z_mem_inv.float() - z_query.detach().float()).pow(2).mean(dim=-1)
        loss = (alpha * recon).sum() / alpha.sum().clamp_min(1.0)
        return loss, {"memory_recon_raw": loss.detach(), "criticality": criticality.detach(), "memory_alpha": alpha.detach()}
