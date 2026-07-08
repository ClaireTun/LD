from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class StudentWorldAdapter(nn.Module):
    """Produces trainable student hidden future/world tokens from VLM cognition tokens.

    Expected input: [B, N, C]. Output tensors are keyed by ``student_world_keys``
    (default: scene/agent/goal), each with shape [B, K, D].  With
    ``student_world_use_cross_attn=True`` learned future queries cross-attend to
    cognition tokens, so H_future is directly conditioned on the VLM hidden state.
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
        student_world_keys: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.student_world_num_tokens = student_world_num_tokens
        self.student_world_dim = student_world_dim
        self.student_world_use_cross_attn = student_world_use_cross_attn
        self.student_world_keys = list(student_world_keys or ["scene", "agent", "goal"])
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
            # self.scene_head = nn.Linear(student_world_dim, out_dim)
            # self.agent_head = nn.Linear(student_world_dim, out_dim)
            # self.goal_head = nn.Linear(student_world_dim, out_dim)
            self.key_heads = nn.ModuleDict({
                key: nn.Linear(student_world_dim, out_dim) for key in self.student_world_keys
            })
        self.output_norm = nn.LayerNorm(student_world_dim)

    def forward(self, cognition_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.core(self.input_norm(self.input_proj(cognition_tokens)))
        if self.student_world_use_cross_attn:
            q = self.future_queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(dtype=x.dtype, device=x.device)
            h, _ = self.future_attn(q, x, x, need_weights=False)
            h = self.output_norm(h + self.future_ffn(h))
            #return {"scene": h, "agent": h, "goal": h}
            return {key: h for key in self.student_world_keys}

        pooled = x.mean(dim=1)

        def _reshape(head):
            y = head(pooled)
            return self.output_norm(y.view(y.shape[0], self.student_world_num_tokens, -1))

        #return {"scene": _reshape(self.scene_head), "agent": _reshape(self.agent_head), "goal": _reshape(self.goal_head)}
        return {key: _reshape(head) for key, head in self.key_heads.items()}

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


class DreamConditionSwapDistillLoss(nn.Module):
    """Dream-only condition-swap verifier for ReCogDrive latent future tokens.

    The teacher dream feature is projected once into the student token space only to
    estimate whether teacher correction is useful.  The returned tensor is always a
    zero loss so this module can be used as a judgment/logging condition without
    adding an extra optimization objective.
    """

    def __init__(
        self,
        student_world_dim: int = 256,
        dream_distill_keys: Optional[Iterable[str]] = None,
        loss_type: str = "mse_plus_cosine",
        teacher_weight: float = 1.0,
        self_weight: float = 0.25,
        teacher_anneal_iters: int = 0,
        teacher_min_weight: float = 0.0,
        utility_margin: float = 0.0,
        ema_decay: float = 0.99,
        detach_teacher: bool = True,
        normalize: bool = True,
        student_world_num_tokens: int = 1,
        projector_type: str = "mlp",
        teacher_latent_dim: int = 256,
        projector_num_heads: int = 4,
        projector_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dream_distill_keys = list(dream_distill_keys or ["dream_scene", "dream_agent"])
        self.teacher_weight = float(teacher_weight)
        self.self_weight = float(self_weight)
        self.teacher_anneal_iters = int(teacher_anneal_iters)
        self.teacher_min_weight = float(teacher_min_weight)
        self.utility_margin = float(utility_margin)
        self.ema_decay = float(ema_decay)
        self.detach_teacher = bool(detach_teacher)
        self.teacher_to_token_projector = nn.ModuleDict({
            key: TeacherToTokenProjector(
                teacher_latent_dim=teacher_latent_dim,
                student_world_dim=student_world_dim,
                num_tokens=student_world_num_tokens,
                projector_type=projector_type,
                num_heads=projector_num_heads,
                dropout=projector_dropout,
            ) for key in self.dream_distill_keys
        })
        self.hidden_loss = HiddenDistillationLoss(loss_type=loss_type, weight=1.0, normalize=normalize)
        self._ema_self_teacher: Dict[str, torch.Tensor] = {}

    def _teacher_weight(self, iteration: int) -> float:
        if self.teacher_anneal_iters <= 0:
            return self.teacher_weight
        progress = min(1.0, max(0.0, float(iteration) / float(self.teacher_anneal_iters)))
        return self.teacher_min_weight + (self.teacher_weight - self.teacher_min_weight) * (1.0 - progress)

    def _update_ema(self, key: str, value: torch.Tensor) -> None:
        value = value.detach()
        prev = self._ema_self_teacher.get(key)
        if prev is None or prev.shape != value.shape or prev.device != value.device:
            self._ema_self_teacher[key] = value.clone()
        else:
            self._ema_self_teacher[key] = prev.mul(self.ema_decay).add(value, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        student_world: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, Optional[torch.Tensor]],
        iteration: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        first_student = next((v for v in student_world.values() if isinstance(v, torch.Tensor)), None)
        device = first_student.device if first_student is not None else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        logs: Dict[str, torch.Tensor] = {
            "loss_dream_condition_swap": zero,
            "dream_condition_swap_candidate_loss": zero,
            "loss_dream_external": zero,
            "loss_dream_self": zero,
            "dream_teacher_weight": torch.tensor(self._teacher_weight(iteration), device=device),
            "dream_teacher_useful_ratio": zero,
            "dream_ema_ready": zero,
        }
        total = zero
        count = 0
        useful_total = zero
        ema_ready_total = zero
        teacher_weight = self._teacher_weight(iteration)

        for key in self.dream_distill_keys:
            s = student_world.get(key)
            t = teacher_outputs.get(key)
            if s is None or t is None:
                continue
            if self.detach_teacher:
                t = t.detach()
            if t.dim() == 2:
                t = t.unsqueeze(1)
            z_teacher = self.teacher_to_token_projector[key](t.to(dtype=s.dtype, device=s.device))
            if z_teacher.shape[1] != s.shape[1]:
                z_teacher = F.interpolate(z_teacher.transpose(1, 2), size=s.shape[1], mode="nearest").transpose(1, 2)
            if z_teacher.shape[-1] != s.shape[-1]:
                z_teacher = TeacherToTokenProjector._resize_last_dim(z_teacher, s.shape[-1])

            ema_target = self._ema_self_teacher.get(key)
            ema_ready = ema_target is not None and ema_target.shape == s.shape and ema_target.device == s.device
            if not ema_ready:
                # Stage 1: teacher prior initializes latent future thought / EMA self-teacher.
                ema_target = z_teacher.detach().clone()
                self._ema_self_teacher[key] = ema_target
            external_loss, _ = self.hidden_loss(s, z_teacher)
            self_loss, _ = self.hidden_loss(s, ema_target.detach())
            teacher_useful = torch.tensor(
                1.0 if (teacher_weight > self.teacher_min_weight and external_loss.detach() <= self_loss.detach() + self.utility_margin) else 0.0,
                device=s.device,
                dtype=s.dtype,
            )
            key_loss = teacher_useful * (teacher_weight * external_loss) + (1.0 - teacher_useful) * (self.self_weight * self_loss)
            total = total + key_loss
            count += 1
            useful_total = useful_total + teacher_useful.detach().float()
            ema_ready_total = ema_ready_total + torch.tensor(1.0 if ema_ready else 0.0, device=device)
            logs["loss_dream_external"] = logs["loss_dream_external"] + external_loss.detach()
            logs["loss_dream_self"] = logs["loss_dream_self"] + self_loss.detach()
            self._update_ema(key, s)

        if count > 0:
            total = total / count
            logs["dream_condition_swap_candidate_loss"] = total.detach()
            logs["loss_dream_external"] = logs["loss_dream_external"] / count
            logs["loss_dream_self"] = logs["loss_dream_self"] / count
            logs["dream_teacher_useful_ratio"] = useful_total / count
            logs["dream_ema_ready"] = ema_ready_total / count
        # The swap module is a verifier/gating signal only.  Do not backpropagate
        # an additional dream-swap objective; keep the original distillation path
        # responsible for optimization.
        return zero, logs

class StructuralWorldDistillLoss(nn.Module):
    def __init__(
        self,
        student_world_dim: int = 256,
        struct_distill_loss_type: str = "mse_plus_cosine",
        struct_distill_weight: float = 1.0,
        scene_distill_weight: float = 1.0,
        agent_distill_weight: float = 1.0,
        goal_distill_weight: float = 1.0,
        dream_scene_distill_weight: float = 1.0,
        dream_agent_distill_weight: float = 1.0,
        struct_distill_keys: Optional[Iterable[str]] = None,
        detach_teacher: bool = True,
        normalize_distill_features: bool = True,
        student_world_num_tokens: int = 1,
        projector_type: str = "mlp",
        teacher_latent_dim: int = 256,
        projector_num_heads: int = 4,
        projector_dropout: float = 0.0,
    ):
        super().__init__()
        #self.key_w = {"scene": scene_distill_weight, "agent": agent_distill_weight, "goal": goal_distill_weight}
        self.distill_keys: List[str] = list(struct_distill_keys or ["scene", "agent", "goal"])
        self.key_w = {
            "scene": scene_distill_weight,
            "agent": agent_distill_weight,
            "goal": goal_distill_weight,
            "dream_scene": dream_scene_distill_weight,
            "dream_agent": dream_agent_distill_weight,
        }
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
            #for key in ["scene", "agent", "goal"]
            for key in self.distill_keys
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
            # "loss_struct_scene": zero,
            # "loss_struct_agent": zero,
            # "loss_struct_goal": zero,
            **{f"loss_struct_{key}": zero for key in self.distill_keys},
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
        #for key in ["scene", "agent", "goal"]:
        for key in self.distill_keys:
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
            lk = lk * self.key_w.get(key, 1.0)
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