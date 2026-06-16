"""End-to-end DriveMem side-branch module."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .dsu import DSU2D, TokenDSU
from .fusion import MemoryFusionGate
from .ifa import InvariantFeatureAdapter, dsu_consistency_loss, orthogonality_loss
from .pmb import PrototypeMemoryBank


class DriveMemModule(nn.Module):
    def __init__(self, input_dim: int, planner_dim: int, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.use_dsu_side_branch = bool(cfg.get("use_dsu_side_branch", True))
        self.use_dsu_main_path = bool(cfg.get("use_dsu_main_path", False))
        if self.use_dsu_main_path:
            raise ValueError("DriveMem does not support use_dsu_main_path=True; DSU must stay on the side branch.")
        self.dsu_type = str(cfg.get("dsu_type", "token"))
        self.detach_side_branch = bool(cfg.get("detach_side_branch", True))
        self.use_ifa = bool(cfg.get("use_ifa", True))
        self.use_pmb = bool(cfg.get("use_pmb", True))
        self.use_memory_fusion = bool(cfg.get("use_memory_fusion", True))
        self.loss_ifa_orth_weight = float(cfg.get("loss_ifa_orth_weight", 0.01))
        self.loss_dsu_consistency_weight = float(cfg.get("loss_dsu_consistency_weight", 0.05))
        self.loss_memory_recon_weight = float(cfg.get("loss_memory_recon_weight", 0.1))
        self.loss_memory_refine_weight = float(cfg.get("loss_memory_refine_weight", 0.0))
        self.loss_memory_entropy_weight = float(cfg.get("loss_memory_entropy_weight", 0.0))
        self.loss_memory_diversity_weight = float(cfg.get("loss_memory_diversity_weight", 0.0))
        memory_dim = int(cfg.get("memory_dim", cfg.get("ifa_dim", input_dim)))
        ifa_hidden_dim = int(cfg.get("ifa_hidden_dim", max(input_dim, memory_dim) * 2))
        if self.dsu_type == "2d":
            self.dsu = DSU2D(p=cfg.get("dsu_p", 0.05), eps=cfg.get("dsu_eps", 1e-6), factor=cfg.get("dsu_factor", 0.5))
        else:
            self.dsu = TokenDSU(p=cfg.get("dsu_p", 0.05), eps=cfg.get("dsu_eps", 1e-6), factor=cfg.get("dsu_factor", 0.5))
        self.ifa = InvariantFeatureAdapter(input_dim, memory_dim, ifa_hidden_dim, pooling=cfg.get("ifa_pooling", "mean"), dropout=cfg.get("ifa_dropout", 0.0)) if self.use_ifa else nn.Linear(input_dim, memory_dim)
        self.pmb = PrototypeMemoryBank(
            num_prototypes=int(cfg.get("num_prototypes", 128)),
            memory_dim=memory_dim,
            top_k=int(cfg.get("top_k", 8)),
            temperature=float(cfg.get("temperature", 0.07)),
            normalize_query=bool(cfg.get("normalize_query", True)),
            normalize_memory=bool(cfg.get("normalize_memory", True)),
            eps=float(cfg.get("dsu_eps", 1e-6)),
            memory_weight_mode=str(cfg.get("memory_weight_mode", "soft")),
            typical_critical_gamma=float(cfg.get("typical_critical_gamma", 0.8)),
        ) if self.use_pmb else None
        self.fusion = MemoryFusionGate(
            planner_dim=planner_dim,
            memory_dim=memory_dim,
            gate_init_bias=float(cfg.get("memory_gate_init_bias", -2.0)),
            lambda_mem_init=float(cfg.get("lambda_mem_init", 0.1)),
            train_lambda_mem=bool(cfg.get("train_lambda_mem", True)),
        ) if self.use_memory_fusion else None

    def _query_without_ifa(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim == 3:
            z = z.mean(dim=1)
        if isinstance(self.ifa, nn.Linear):
            z = z.to(device=self.ifa.weight.device, dtype=self.ifa.weight.dtype)
        z_inv = self.ifa(z)
        return z_inv, torch.zeros_like(z_inv)

    def forward(self, z_plan: torch.Tensor, side_features: Optional[torch.Tensor] = None, per_sample_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.enabled:
            return z_plan, {}, {}
        if z_plan.ndim != 3:
            raise ValueError(f"DriveMem expects z_plan [B, N, D], got {tuple(z_plan.shape)}")
        z_side_clean = side_features if side_features is not None else z_plan
        if z_side_clean.ndim != 3:
            raise ValueError(f"DriveMem online hidden-state side branch expects [B, N, D], got {tuple(z_side_clean.shape)}")
        if z_side_clean.shape[0] != z_plan.shape[0]:
            raise ValueError("DriveMem z_side_clean and z_plan batch sizes must match")
        if self.detach_side_branch:
            z_side_clean = z_side_clean.detach()
        z_side_dsu = z_side_clean
        if self.training and self.use_dsu_side_branch:
            z_side_dsu = self.dsu(z_side_clean)
        if self.use_ifa:
            z_inv_clean, z_spu_clean = self.ifa(z_side_clean)
            z_inv_dsu, z_spu_dsu = self.ifa(z_side_dsu) if (self.training and self.use_dsu_side_branch) else (z_inv_clean, z_spu_clean)
        else:
            z_inv_clean, z_spu_clean = self._query_without_ifa(z_side_clean)
            z_inv_dsu, z_spu_dsu = self._query_without_ifa(z_side_dsu) if (self.training and self.use_dsu_side_branch) else (z_inv_clean, z_spu_clean)
        z_query = z_inv_dsu if (self.training and self.use_dsu_side_branch) else z_inv_clean
        losses: Dict[str, torch.Tensor] = {}
        logs: Dict[str, torch.Tensor] = {
            "z_inv_norm": z_query.detach().float().norm(dim=-1).mean(),
        }
        if self.loss_ifa_orth_weight > 0:
            losses["loss_ifa_orth"] = self.loss_ifa_orth_weight * orthogonality_loss(z_query, z_spu_dsu)
        if self.training and self.use_dsu_side_branch and self.loss_dsu_consistency_weight > 0:
            losses["loss_dsu_consistency"] = self.loss_dsu_consistency_weight * dsu_consistency_loss(z_inv_dsu, z_inv_clean)
        if self.pmb is not None:
            z_mem_inv, assignment, pmb_logs = self.pmb(z_query)
            logs.update(pmb_logs)
            if self.training and self.loss_memory_recon_weight > 0:
                recon_loss, recon_logs = self.pmb.loss(z_query, z_mem_inv, assignment, per_sample_loss=per_sample_loss)
                losses["loss_memory_recon"] = self.loss_memory_recon_weight * recon_loss
                logs.update(recon_logs)
        else:
            z_mem_inv = z_query
        if self.fusion is not None:
            z_fused, fusion_logs = self.fusion(z_plan, z_mem_inv)
            logs.update(fusion_logs)
        else:
            z_fused = z_plan
        return z_fused, losses, logs
