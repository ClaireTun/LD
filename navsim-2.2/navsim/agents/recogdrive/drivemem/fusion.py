"""Residual planner-side memory fusion for DriveMem."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class MemoryFusionGate(nn.Module):
    def __init__(self, planner_dim: int, memory_dim: int, gate_init_bias: float = -2.0, lambda_mem_init: float = 0.1, train_lambda_mem: bool = True) -> None:
        super().__init__()
        self.mem_to_plan_proj = nn.Linear(memory_dim, planner_dim)
        self.gate = nn.Sequential(
            nn.Linear(planner_dim * 2, planner_dim),
            nn.GELU(),
            nn.Linear(planner_dim, planner_dim),
        )
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)
        lambda_tensor = torch.tensor(float(lambda_mem_init))
        if train_lambda_mem:
            self.lambda_mem = nn.Parameter(lambda_tensor)
        else:
            self.register_buffer("lambda_mem", lambda_tensor)

    def forward(self, z_plan: torch.Tensor, z_mem_inv: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z_plan.ndim != 3:
            raise ValueError(f"MemoryFusionGate expects z_plan [B, N, D], got {tuple(z_plan.shape)}")
        if z_mem_inv.ndim != 2 or z_mem_inv.shape[0] != z_plan.shape[0]:
            raise ValueError(f"MemoryFusionGate expects z_mem_inv [B, Dm], got {tuple(z_mem_inv.shape)} for z_plan {tuple(z_plan.shape)}")
        z_mem_plan = self.mem_to_plan_proj(z_mem_inv.to(dtype=self.mem_to_plan_proj.weight.dtype, device=self.mem_to_plan_proj.weight.device))
        z_mem_plan = z_mem_plan.to(dtype=z_plan.dtype, device=z_plan.device)
        z_mem_tokens = z_mem_plan[:, None, :].expand_as(z_plan)
        gate_in = torch.cat([z_plan, z_mem_tokens], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in.to(dtype=self.gate[0].weight.dtype, device=self.gate[0].weight.device))).to(dtype=z_plan.dtype, device=z_plan.device)
        z_fused = z_plan + self.lambda_mem.to(dtype=z_plan.dtype, device=z_plan.device) * gate * z_mem_tokens
        logs = {
            "z_mem_plan": z_mem_plan.detach(),
            "memory_gate_mean": gate.detach().float().mean(),
            "memory_gate_std": gate.detach().float().std(unbiased=False),
            "lambda_mem": self.lambda_mem.detach().float(),
        }
        return z_fused, logs
