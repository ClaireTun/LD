from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class StudentWorldAdapter(nn.Module):
    """Produces trainable student hidden future/world tokens from VLM cognition tokens.

    Expected input: [B, N, C]. Output scene/agent/goal tensors: [B, K, D].
    With ``student_world_use_cross_attn=True`` learned future queries cross-attend
    to cognition tokens, so H_future is directly conditioned on the VLM hidden state.
    """

    def __init__(
        self,
        input_dim: int,
        student_world_dim: int = 256,
        student_world_num_tokens: int = 1,
        student_world_use_mlp: bool = True,
        student_world_dropout: float = 0.0,
        student_world_use_cross_attn: bool = False,
        student_world_num_heads: int = 4,
    ):
        super().__init__()
        self.student_world_num_tokens = student_world_num_tokens
        self.student_world_dim = student_world_dim
        self.student_world_use_cross_attn = student_world_use_cross_attn
        self.input_proj = nn.Linear(input_dim, student_world_dim)
        self.input_norm = nn.LayerNorm(student_world_dim)
        self.core = nn.Sequential(
            nn.LayerNorm(student_world_dim),
            nn.GELU(),
            nn.Dropout(student_world_dropout),
            nn.Linear(student_world_dim, student_world_dim),
        ) if student_world_use_mlp else nn.Identity()
        if student_world_use_cross_attn:
            self.future_queries = nn.Parameter(torch.randn(student_world_num_tokens, student_world_dim) * 0.02)
            self.future_attn = nn.MultiheadAttention(
                embed_dim=student_world_dim,
                num_heads=student_world_num_heads,
                dropout=student_world_dropout,
                batch_first=True,
            )
            self.future_ffn = nn.Sequential(
                nn.LayerNorm(student_world_dim),
                nn.Linear(student_world_dim, student_world_dim * 4),
                nn.GELU(),
                nn.Dropout(student_world_dropout),
                nn.Linear(student_world_dim * 4, student_world_dim),
            )
        else:
            out_dim = student_world_dim * student_world_num_tokens
            self.scene_head = nn.Linear(student_world_dim, out_dim)
            self.agent_head = nn.Linear(student_world_dim, out_dim)
            self.goal_head = nn.Linear(student_world_dim, out_dim)
        self.output_norm = nn.LayerNorm(student_world_dim)

    def forward(self, cognition_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.core(self.input_norm(self.input_proj(cognition_tokens)))
        if self.student_world_use_cross_attn:
            q = self.future_queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(dtype=x.dtype, device=x.device)
            h, _ = self.future_attn(q, x, x, need_weights=False)
            h = self.output_norm(h + self.future_ffn(h))
            return {"scene": h, "agent": h, "goal": h}

        pooled = x.mean(dim=1)

        def _reshape(head):
            y = head(pooled)
            return self.output_norm(y.view(y.shape[0], self.student_world_num_tokens, -1))

        return {"scene": _reshape(self.scene_head), "agent": _reshape(self.agent_head), "goal": _reshape(self.goal_head)}


class TeacherToTokenProjector(nn.Module):
    """Project SGDrive teacher latent [B, Nt, Dt] into K student tokens [B, K, Ds]."""

    def __init__(
        self,
        teacher_latent_dim: int = 256,
        student_world_dim: int = 256,
        num_tokens: int = 1,
        projector_type: str = "mlp",
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.teacher_latent_dim = teacher_latent_dim
        self.student_world_dim = student_world_dim
        self.num_tokens = num_tokens
        self.projector_type = projector_type
        self.in_norm = nn.LayerNorm(teacher_latent_dim)
        self.out_norm = nn.LayerNorm(student_world_dim)
        if projector_type == "linear_pool":
            self.proj = nn.Linear(teacher_latent_dim, student_world_dim)
            self.token_offsets = nn.Parameter(torch.randn(num_tokens, student_world_dim) * 0.02)
        elif projector_type == "mlp":
            self.proj = nn.Sequential(
                nn.LayerNorm(teacher_latent_dim),
                nn.Linear(teacher_latent_dim, student_world_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(student_world_dim * 2, student_world_dim),
                nn.LayerNorm(student_world_dim),
            )
            self.token_offsets = nn.Parameter(torch.randn(num_tokens, student_world_dim) * 0.02)
        elif projector_type == "cross_attn":
            self.teacher_proj = nn.Sequential(
                nn.LayerNorm(teacher_latent_dim),
                nn.Linear(teacher_latent_dim, student_world_dim),
                nn.LayerNorm(student_world_dim),
            )
            self.teacher_queries = nn.Parameter(torch.randn(num_tokens, student_world_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(student_world_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.LayerNorm(student_world_dim),
                nn.Linear(student_world_dim, student_world_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(student_world_dim * 4, student_world_dim),
            )
        else:
            raise ValueError(f"Unsupported projector_type={projector_type}")

    @staticmethod
    def _resize_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        flat = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        flat = F.adaptive_avg_pool1d(flat, dim).squeeze(1)
        return flat.reshape(*x.shape[:-1], dim)

    def forward(self, teacher_latent: torch.Tensor) -> torch.Tensor:
        if teacher_latent.dim() == 2:
            teacher_latent = teacher_latent.unsqueeze(1)
        teacher_latent = self._resize_last_dim(teacher_latent, self.teacher_latent_dim)
        if self.projector_type in ("linear_pool", "mlp"):
            pooled = teacher_latent.mean(dim=1)
            z = self.proj(pooled).unsqueeze(1).expand(-1, self.num_tokens, -1)
            z = z + self.token_offsets.unsqueeze(0).to(z.dtype)
            return self.out_norm(z)
        memory = self.teacher_proj(teacher_latent)
        q = self.teacher_queries.unsqueeze(0).expand(memory.shape[0], -1, -1).to(dtype=memory.dtype, device=memory.device)
        z, _ = self.cross_attn(q, memory, memory, need_weights=False)
        return self.out_norm(z + self.ffn(z))


class HiddenDistillationLoss(nn.Module):
    def __init__(self, loss_type: str = "mse_plus_cosine", weight: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.normalize = normalize

    def forward(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h_future.shape[1] != z_teacher.shape[1]:
            z_teacher = F.interpolate(z_teacher.transpose(1, 2), size=h_future.shape[1], mode="nearest").transpose(1, 2)
        if h_future.shape[-1] != z_teacher.shape[-1]:
            z_teacher = TeacherToTokenProjector._resize_last_dim(z_teacher, h_future.shape[-1])
        h_for_mse = F.normalize(h_future, dim=-1) if self.normalize else h_future
        z_for_mse = F.normalize(z_teacher, dim=-1) if self.normalize else z_teacher
        mse = F.mse_loss(h_for_mse, z_for_mse, reduction="mean")
        cos_sim = F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        if self.loss_type == "mse_norm":
            loss = mse
        elif self.loss_type == "cosine":
            loss = cos_loss
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(h_for_mse, z_for_mse, reduction="mean")
        elif self.loss_type == "mse_plus_cosine":
            loss = mse + 0.5 * cos_loss
        else:
            raise ValueError(f"Unsupported hidden distill loss type: {self.loss_type}")
        logs = {
            "loss_hidden": loss.detach(),
            "loss_hidden_mse": mse.detach(),
            "loss_hidden_cos": cos_loss.detach(),
            "hidden_cos_sim": cos_sim.detach(),
            "H_future_norm": h_future.detach().float().norm(dim=-1).mean(),
            "Z_teacher_norm": z_teacher.detach().float().norm(dim=-1).mean(),
        }
        return loss * self.weight, logs


class StructuralWorldDistillLoss(nn.Module):
    def __init__(
        self,
        student_world_dim: int = 256,
        struct_distill_loss_type: str = "mse_plus_cosine",
        struct_distill_weight: float = 1.0,
        scene_distill_weight: float = 1.0,
        agent_distill_weight: float = 1.0,
        goal_distill_weight: float = 1.0,
        detach_teacher: bool = True,
        normalize_distill_features: bool = True,
        student_world_num_tokens: int = 1,
        projector_type: str = "mlp",
        teacher_latent_dim: int = 256,
        projector_num_heads: int = 4,
        projector_dropout: float = 0.0,
    ):
        super().__init__()
        self.key_w = {"scene": scene_distill_weight, "agent": agent_distill_weight, "goal": goal_distill_weight}
        self.detach_teacher = detach_teacher
        self.student_world_dim = student_world_dim
        self.teacher_to_token_projector = nn.ModuleDict({
            key: TeacherToTokenProjector(
                teacher_latent_dim=teacher_latent_dim,
                student_world_dim=student_world_dim,
                num_tokens=student_world_num_tokens,
                projector_type=projector_type,
                num_heads=projector_num_heads,
                dropout=projector_dropout,
            )
            for key in ["scene", "agent", "goal"]
        })
        self.hidden_loss = HiddenDistillationLoss(
            loss_type=struct_distill_loss_type,
            weight=struct_distill_weight,
            normalize=normalize_distill_features,
        )

    def forward(self, student_world: Dict[str, torch.Tensor], teacher_outputs: Dict[str, Optional[torch.Tensor]]):
        first_student = next((v for v in student_world.values() if isinstance(v, torch.Tensor)), None)
        device = first_student.device if first_student is not None else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        logs: Dict[str, torch.Tensor] = {
            "loss_struct_scene": zero,
            "loss_struct_agent": zero,
            "loss_struct_goal": zero,
            "loss_struct_total": zero,
            "loss_hidden": zero,
            "loss_hidden_mse": zero,
            "loss_hidden_cos": zero,
            "hidden_cos_sim": zero,
            "H_future_norm": zero,
            "Z_teacher_norm": zero,
            "teacher_latent_norm": zero,
        }
        total = zero
        count = 0
        for key in ["scene", "agent", "goal"]:
            t = teacher_outputs.get(key, None)
            s = student_world.get(key, None)
            if t is None or s is None:
                continue
            if self.detach_teacher:
                t = t.detach()
            if t.dim() == 2:
                t = t.unsqueeze(1)
            z = self.teacher_to_token_projector[key](t.to(dtype=s.dtype, device=s.device))
            lk, lk_logs = self.hidden_loss(s, z)
            lk = lk * self.key_w[key]
            logs[f"loss_struct_{key}"] = lk.detach()
            total = total + lk
            count += 1
            for log_key, value in lk_logs.items():
                logs[log_key] = logs[log_key] + value.detach()
            logs["teacher_latent_norm"] = logs["teacher_latent_norm"] + t.detach().float().norm(dim=-1).mean().to(device)
        if count > 0:
            total = total / count
            for log_key in ["loss_hidden", "loss_hidden_mse", "loss_hidden_cos", "hidden_cos_sim", "H_future_norm", "Z_teacher_norm", "teacher_latent_norm"]:
                logs[log_key] = logs[log_key] / count
        logs["loss_struct_total"] = total.detach()
        return total, logs