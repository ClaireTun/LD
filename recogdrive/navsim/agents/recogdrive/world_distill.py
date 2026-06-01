from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class StudentWorldAdapter(nn.Module):
    """Lightweight adapter producing structural planning tokens from cognition tokens.

    Expected input shape: [B, N, C]
    Output tensors (scene/agent/goal): [B, T, D]
    """

    def __init__(
        self,
        input_dim: int,
        student_world_dim: int = 256,
        student_world_num_tokens: int = 1,
        student_world_use_mlp: bool = True,
        student_world_dropout: float = 0.0,
    ):
        super().__init__()
        self.student_world_num_tokens = student_world_num_tokens
        self.input_proj = nn.Linear(input_dim, student_world_dim)
        self.core = nn.Sequential(
            nn.LayerNorm(student_world_dim),
            nn.GELU(),
            nn.Dropout(student_world_dropout),
            nn.Linear(student_world_dim, student_world_dim),
        ) if student_world_use_mlp else nn.Identity()
        out_dim = student_world_dim * student_world_num_tokens
        self.scene_head = nn.Linear(student_world_dim, out_dim)
        self.agent_head = nn.Linear(student_world_dim, out_dim)
        self.goal_head = nn.Linear(student_world_dim, out_dim)

    def forward(self, cognition_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(cognition_tokens)   # [B, N, D]
        x = self.core(x)
        pooled = x.mean(dim=1)                  # [B, D]

        def _reshape(head):
            y = head(pooled)
            return y.view(y.shape[0], self.student_world_num_tokens, -1)

        return {
            "scene": _reshape(self.scene_head),
            "agent": _reshape(self.agent_head),
            "goal": _reshape(self.goal_head),
        }


class StructuralWorldDistillLoss(nn.Module):
    def __init__(
        self,
        student_world_dim: int = 256,
        struct_distill_loss_type: str = "cosine",
        struct_distill_weight: float = 1.0,
        scene_distill_weight: float = 1.0,
        agent_distill_weight: float = 1.0,
        goal_distill_weight: float = 1.0,
        detach_teacher: bool = True,
        normalize_distill_features: bool = True,
    ):
        super().__init__()
        self.loss_type = struct_distill_loss_type
        self.struct_w = struct_distill_weight
        self.key_w = {
            "scene": scene_distill_weight,
            "agent": agent_distill_weight,
            "goal": goal_distill_weight,
        }
        self.detach_teacher = detach_teacher
        self.normalize = normalize_distill_features
        self.student_world_dim = student_world_dim

    def _calc(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(student, teacher)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(student, teacher)
        # cosine
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()

    def _match_feature_dim(self, teacher: torch.Tensor, target_dim: int) -> torch.Tensor:
        """DDP-safe, parameter-free teacher feature alignment.

        LazyLinear parameters are initialized only on first forward, which breaks
        DistributedDataParallel setup. This deterministic resize keeps the
        distillation loss shape-compatible without uninitialized parameters.
        """
        if teacher.shape[-1] == target_dim:
            return teacher
        teacher_2d = teacher.reshape(-1, teacher.shape[-1]).unsqueeze(1)
        teacher_2d = F.adaptive_avg_pool1d(teacher_2d, target_dim).squeeze(1)
        return teacher_2d.reshape(*teacher.shape[:-1], target_dim)

    def forward(self, student_world: Dict[str, torch.Tensor], teacher_outputs: Dict[str, Optional[torch.Tensor]]):
        first_student = next((v for v in student_world.values() if isinstance(v, torch.Tensor)), None)
        device = first_student.device if first_student is not None else torch.device("cpu")
        logs = {
            "loss_struct_scene": torch.tensor(0.0, device=device),
            "loss_struct_agent": torch.tensor(0.0, device=device),
            "loss_struct_goal": torch.tensor(0.0, device=device),
            "loss_struct_total": torch.tensor(0.0, device=device),
        }
        total = logs["loss_struct_total"]
        for key in ["scene", "agent", "goal"]:
            t = teacher_outputs.get(key, None)
            s = student_world.get(key, None)
            if t is None or s is None:
                continue
            if self.detach_teacher:
                t = t.detach()
            # [B, Nt, Ct] -> [B, Nt, D] without lazy/trainable parameters.
            if t.dim() == 2:
                t = t.unsqueeze(1)
            t = self._match_feature_dim(t, s.shape[-1])
            if self.normalize:
                t = F.normalize(t, dim=-1)
                s = F.normalize(s, dim=-1)
            lk = self._calc(s, t) * self.key_w[key]
            logs[f"loss_struct_{key}"] = lk
            total = total + lk
        logs["loss_struct_total"] = total * self.struct_w
        return logs["loss_struct_total"], logs
