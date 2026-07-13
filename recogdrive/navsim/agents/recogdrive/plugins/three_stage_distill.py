# from __future__ import annotations

# import math
# from dataclasses import dataclass
# from typing import Any, Dict, Mapping, Optional, Tuple

# import torch
# from torch import nn
# import torch.nn.functional as F


# @dataclass
# class ThreeStageDistillConfig:
#     use_three_stage_distill: bool = False
#     stage1_epochs: int = 2
#     stage1_iters: int = -1
#     stage2_end_epochs: int = 7
#     lambda_init: float = 1.0
#     lambda_corr: float = 1.0
#     lambda_cons: float = 0.1
#     lambda_teacher_min: float = 0.0
#     teacher_decay_type: str = "linear"
#     beta_cos: float = 0.5
#     use_soft_intervention: bool = True
#     use_uncertainty: bool = False
#     use_self_consistency: bool = False
#     use_counterfactual_gain: bool = False
#     use_failure_type_attribution: bool = False
#     tau_f: float = 0.5
#     tau_d: float = 0.5
#     tau_u: float = 0.1
#     soft_a: float = 1.0
#     soft_b: float = 1.0
#     soft_c: float = 1.0
#     soft_d: float = 1.0
#     soft_threshold: float = 0.0
#     alpha_l1: float = 1.0
#     alpha_collision: float = 1.0
#     alpha_offroad: float = 1.0
#     alpha_route: float = 1.0
#     alpha_ttc: float = 1.0
#     eps: float = 1e-6
#     debug_sanity: bool = True


# def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
#     if cfg is None:
#         return default
#     if isinstance(cfg, Mapping):
#         return cfg.get(name, default)
#     return getattr(cfg, name, default)


# def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
#     if value is None:
#         return None
#     if isinstance(value, torch.Tensor):
#         return value.to(device=device, dtype=dtype)
#     try:
#         return torch.as_tensor(value, device=device, dtype=dtype)
#     except (TypeError, ValueError):
#         return None


# class ThreeStageImaginationDistiller(nn.Module):
#     """Plugin-style controller for LatentSight three-stage imagination distillation.

#     The module is intentionally independent from the planner forward. It consumes
#     tensors already produced by the student, teacher projector, and optional debug
#     forwards, then returns additive losses and scalar logs.
#     """

#     def __init__(self, cfg: Optional[Any] = None, **kwargs: Any) -> None:
#         super().__init__()
#         merged: Dict[str, Any] = {}
#         for field_name in ThreeStageDistillConfig.__dataclass_fields__:
#             merged[field_name] = _cfg_get(cfg, field_name, getattr(ThreeStageDistillConfig(), field_name))
#         merged.update(kwargs)
#         self.cfg = ThreeStageDistillConfig(**merged)
#         self._printed_sanity = False

#     def current_stage(self, epoch: int = 0, iteration: int = 0) -> str:
#         if self.cfg.stage1_iters is not None and self.cfg.stage1_iters >= 0 and iteration < self.cfg.stage1_iters:
#             return "init"
#         if epoch < self.cfg.stage1_epochs:
#             return "init"
#         if epoch < self.cfg.stage2_end_epochs:
#             return "corrective"
#         return "self_evolving"

#     def teacher_lambda(self, epoch: int = 0) -> float:
#         if epoch < self.cfg.stage2_end_epochs:
#             return float(self.cfg.lambda_corr)
#         span = max(1, self.cfg.stage2_end_epochs - self.cfg.stage1_epochs)
#         progress = min(1.0, max(0.0, (epoch - self.cfg.stage2_end_epochs + 1) / span))
#         if self.cfg.teacher_decay_type == "cosine":
#             factor = 0.5 * (1.0 + math.cos(math.pi * progress))
#         else:
#             factor = 1.0 - progress
#         return float(self.cfg.lambda_teacher_min + (self.cfg.lambda_corr - self.cfg.lambda_teacher_min) * factor)

#     def align_tokens(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         if source.shape[1] != target.shape[1]:
#             source = F.interpolate(source.transpose(1, 2), size=target.shape[1], mode="nearest").transpose(1, 2)
#         if source.shape[-1] != target.shape[-1]:
#             if source.shape[-1] > target.shape[-1]:
#                 source = source[..., : target.shape[-1]]
#             else:
#                 pad = target.shape[-1] - source.shape[-1]
#                 source = F.pad(source, (0, pad))
#         return source

#     def dist_per_sample(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
#         z_teacher = self.align_tokens(z_teacher, h_future)
#         h_norm = F.normalize(h_future.float(), dim=-1)
#         z_norm = F.normalize(z_teacher.float(), dim=-1)
#         mse = (h_norm - z_norm).pow(2).mean(dim=(1, 2))
#         cos = 1.0 - F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1).mean(dim=1)
#         return mse + self.cfg.beta_cos * cos

#     def dist_mean(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         z_teacher = self.align_tokens(z_teacher, h_future)
#         h_norm = F.normalize(h_future.float(), dim=-1)
#         z_norm = F.normalize(z_teacher.float(), dim=-1)
#         mse = F.mse_loss(h_norm, z_norm, reduction="mean")
#         cos_sim = F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1).mean()
#         cos_loss = 1.0 - cos_sim
#         loss = mse + self.cfg.beta_cos * cos_loss
#         logs = {
#             "loss_hidden_mse": mse.detach(),
#             "loss_hidden_cos": cos_loss.detach(),
#             "hidden_cos_sim": cos_sim.detach(),
#             "H_future_norm": h_future.detach().float().norm(dim=-1).mean(),
#             "Z_teacher_norm": z_teacher.detach().float().norm(dim=-1).mean(),
#         }
#         return loss, logs

#     def _per_sample_l1(self, pred: Optional[torch.Tensor], gt: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
#         if pred is None or gt is None:
#             return torch.zeros(batch_size, device=device)
#         gt = gt.to(device=pred.device, dtype=pred.dtype)
#         if pred.shape != gt.shape:
#             min_dims = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
#             pred = pred[tuple(slice(0, d) for d in min_dims)]
#             gt = gt[tuple(slice(0, d) for d in min_dims)]
#         return (pred.float() - gt.float()).abs().flatten(start_dim=1).mean(dim=1)

#     def failure_score(
#         self,
#         traj_pred: Optional[torch.Tensor],
#         traj_gt: Optional[torch.Tensor],
#         extras: Optional[Mapping[str, Any]],
#         batch_size: int,
#         device: torch.device,
#     ) -> torch.Tensor:
#         score = self.cfg.alpha_l1 * self._per_sample_l1(traj_pred, traj_gt, batch_size, device)
#         extras = extras or {}
#         optional_terms = [
#             ("collision_risk", self.cfg.alpha_collision),
#             ("offroad_risk", self.cfg.alpha_offroad),
#             ("route_deviation", self.cfg.alpha_route),
#             ("ttc_violation", self.cfg.alpha_ttc),
#         ]
#         for key, alpha in optional_terms:
#             value = _as_tensor(extras.get(key), device, score.dtype)
#             if value is not None:
#                 score = score + alpha * value.flatten(start_dim=1).mean(dim=1) if value.ndim > 1 else score + alpha * value.flatten()
#         return score

#     def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.float()
#         if x.numel() <= 1:
#             return torch.zeros_like(x)
#         std = x.std(unbiased=False)
#         if not torch.isfinite(std) or std < self.cfg.eps:
#             return torch.zeros_like(x)
#         return (x - x.mean()) / (std + self.cfg.eps)

#     def intervention_weight(self, failure: torch.Tensor, discrepancy: torch.Tensor, uncertainty: torch.Tensor, gain: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         if self.cfg.use_soft_intervention:
#             logits = (
#                 self.cfg.soft_a * self._batch_norm(failure)
#                 + self.cfg.soft_b * self._batch_norm(discrepancy)
#                 + self.cfg.soft_c * self._batch_norm(uncertainty)
#                 + self.cfg.soft_d * self._batch_norm(gain)
#                 - self.cfg.soft_threshold
#             )
#             weights = torch.sigmoid(logits)
#             selected = (weights > 0.5).float().mean()
#             logs = {"selected_sample_ratio": selected.detach(), "intervention_ratio": selected.detach()}
#         else:
#             mask = (failure > self.cfg.tau_f) & ((discrepancy > self.cfg.tau_d) | (uncertainty > self.cfg.tau_u))
#             weights = mask.to(dtype=failure.dtype)
#             logs = {"selected_sample_ratio": weights.mean().detach(), "intervention_ratio": weights.mean().detach()}
#         return weights, logs

#     def weighted_distill(self, h_future: torch.Tensor, z_teacher: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#         per_sample = self.dist_per_sample(h_future, z_teacher)
#         weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights)).to(per_sample.dtype)
#         denom = weights.sum().clamp_min(self.cfg.eps)
#         if weights.sum().detach() <= self.cfg.eps:
#             return per_sample.sum() * 0.0
#         return (per_sample * weights).sum() / denom

#     def compute_consistency_loss(
#         self,
#         h_future: torch.Tensor,
#         h_future_ref: Optional[torch.Tensor],
#         mask_or_weight: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if h_future_ref is None:
#             return h_future.sum() * 0.0
#         h_future_ref = self.align_tokens(h_future_ref.detach(), h_future)
#         per_sample = self.dist_per_sample(h_future, h_future_ref)
#         if mask_or_weight is None:
#             return per_sample.mean()
#         weight = mask_or_weight.to(device=per_sample.device, dtype=per_sample.dtype)
#         if weight.sum().detach() <= self.cfg.eps:
#             return per_sample.sum() * 0.0
#         return (per_sample * weight).sum() / weight.sum().clamp_min(self.cfg.eps)

#     def sanity_check(self, tensors: Mapping[str, Optional[torch.Tensor]], logs: Mapping[str, torch.Tensor], stage: str) -> None:
#         if self._printed_sanity or not self.cfg.debug_sanity:
#             return
#         self._printed_sanity = True
#         h = tensors.get("H_future")
#         z = tensors.get("Z_teacher")
#         pred = tensors.get("traj_pred")
#         gt = tensors.get("traj_gt")
#         def _shape(x: Optional[torch.Tensor]) -> str:
#             return "None" if x is None else str(tuple(x.shape))
#         def _stats(name: str, x: Optional[torch.Tensor]) -> str:
#             if x is None:
#                 return f"{name}=None"
#             xf = x.detach().float()
#             return f"{name}: mean={xf.mean().item():.6f} std={xf.std(unbiased=False).item():.6f} norm={xf.norm(dim=-1).mean().item():.6f}"
#         hidden_finite = bool(torch.isfinite(logs.get("loss_hidden_init", torch.tensor(0.0, device=h.device if h is not None else "cpu"))).all().item())
#         weight = logs.get("intervention_weight_mean")
#         weight_ok = weight is None or bool(torch.isfinite(weight).all().item())
#         print(
#             "[LatentSight][ThreeStage][Sanity] "
#             f"stage={stage} H_future_shape={_shape(h)} Z_teacher_shape={_shape(z)} "
#             f"traj_pred_shape={_shape(pred)} traj_gt_shape={_shape(gt)} "
#             f"{_stats('H_future', h)} {_stats('Z_teacher', z)} "
#             f"hidden_loss_finite={hidden_finite} intervention_weight_ok={weight_ok}"
#         )

#     def forward(
#         self,
#         *,
#         h_future: Optional[torch.Tensor],
#         z_teacher: Optional[torch.Tensor],
#         traj_pred: Optional[torch.Tensor] = None,
#         traj_gt: Optional[torch.Tensor] = None,
#         h_future_aug: Optional[torch.Tensor] = None,
#         h_future_ref: Optional[torch.Tensor] = None,
#         extras: Optional[Mapping[str, Any]] = None,
#         epoch: int = 0,
#         iteration: int = 0,
#         teacher_enabled: bool = True,
#         teacher_requires_grad_count: int = 0,
#     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         device = h_future.device if h_future is not None else (z_teacher.device if z_teacher is not None else torch.device("cpu"))
#         zero = torch.tensor(0.0, device=device)
#         stage = self.current_stage(epoch, iteration)
#         logs: Dict[str, torch.Tensor] = {
#             "loss_hidden_init": zero,
#             "loss_corrective": zero,
#             "loss_consistency": zero,
#             "lambda_teacher": torch.tensor(self.teacher_lambda(epoch), device=device),
#             "current_distill_stage": torch.tensor({"init": 1.0, "corrective": 2.0, "self_evolving": 3.0}[stage], device=device),
#             "current_distill_stage_init": torch.tensor(1.0 if stage == "init" else 0.0, device=device),
#             "current_distill_stage_corrective": torch.tensor(1.0 if stage == "corrective" else 0.0, device=device),
#             "current_distill_stage_self_evolving": torch.tensor(1.0 if stage == "self_evolving" else 0.0, device=device),
#             "use_three_stage_distill": torch.tensor(1.0 if self.cfg.use_three_stage_distill else 0.0, device=device),
#             "use_counterfactual_gain": torch.tensor(1.0 if self.cfg.use_counterfactual_gain else 0.0, device=device),
#             "use_uncertainty": torch.tensor(1.0 if self.cfg.use_uncertainty else 0.0, device=device),
#             "use_self_consistency": torch.tensor(1.0 if self.cfg.use_self_consistency else 0.0, device=device),
#             "teacher_enabled": torch.tensor(1.0 if teacher_enabled else 0.0, device=device),
#             "teacher_requires_grad_count": torch.tensor(float(teacher_requires_grad_count), device=device),
#             "failure_score_mean": zero,
#             "discrepancy_score_mean": zero,
#             "uncertainty_score_mean": zero,
#             "intervention_weight_mean": zero,
#             "intervention_weight_max": zero,
#             "intervention_ratio": zero,
#             "selected_sample_ratio": zero,
#             "hidden_cos_sim": zero,
#             "H_future_norm": zero,
#             "Z_teacher_norm": zero,
#         }
#         losses = {"loss_hidden": zero, "loss_corrective": zero, "loss_consistency": zero}

#         if not self.cfg.use_three_stage_distill or not teacher_enabled or h_future is None or z_teacher is None:
#             return losses, logs

#         z_teacher = self.align_tokens(z_teacher.to(device=h_future.device, dtype=h_future.dtype), h_future)
#         hidden_loss, hidden_logs = self.dist_mean(h_future, z_teacher)
#         logs.update(hidden_logs)
#         logs["loss_hidden_init"] = hidden_loss.detach()
#         discrepancy = self.dist_per_sample(h_future.detach(), z_teacher.detach())
#         batch_size = h_future.shape[0]

#         if stage == "init":
#             losses["loss_hidden"] = self.cfg.lambda_init * hidden_loss
#         else:
#             failure = self.failure_score(traj_pred, traj_gt, extras, batch_size, h_future.device)
#             if self.cfg.use_uncertainty and h_future_aug is not None:
#                 uncertainty = self.dist_per_sample(h_future.detach(), h_future_aug.detach())
#             else:
#                 uncertainty = torch.zeros_like(failure)
#             gain = torch.zeros_like(failure)
#             weights, weight_logs = self.intervention_weight(failure, discrepancy, uncertainty, gain)
#             corr = self.weighted_distill(h_future, z_teacher, weights)
#             logs.update(weight_logs)
#             logs["failure_score_mean"] = failure.detach().mean()
#             logs["discrepancy_score_mean"] = discrepancy.detach().mean()
#             logs["uncertainty_score_mean"] = uncertainty.detach().mean()
#             logs["intervention_weight_mean"] = weights.detach().mean()
#             logs["intervention_weight_max"] = weights.detach().max() if weights.numel() else zero
#             if stage == "corrective":
#                 losses["loss_corrective"] = self.cfg.lambda_corr * corr
#             else:
#                 teacher_weight = self.teacher_lambda(epoch)
#                 losses["loss_corrective"] = teacher_weight * corr
#                 low_failure_weight = (1.0 - weights).detach()
#                 ref = h_future_ref if h_future_ref is not None else (h_future_aug if self.cfg.use_uncertainty else None)
#                 cons = self.compute_consistency_loss(h_future, ref, low_failure_weight) if self.cfg.use_self_consistency else h_future.sum() * 0.0
#                 losses["loss_consistency"] = self.cfg.lambda_cons * cons
#                 logs["loss_consistency"] = cons.detach()
#             logs["loss_corrective"] = corr.detach()

#         self.sanity_check({"H_future": h_future, "Z_teacher": z_teacher, "traj_pred": traj_pred, "traj_gt": traj_gt}, logs, stage)
#         return losses, logs


# def build_imagination_distiller(cfg: Optional[Any] = None, **kwargs: Any) -> ThreeStageImaginationDistiller:
#     return ThreeStageImaginationDistiller(cfg, **kwargs)


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AuxiliaryPlanningProbe(nn.Module):
    """Training-only probe: predicts a trajectory from detached decision hidden + live future slots."""

    def __init__(
        self,
        decision_hidden_dim: int = 1536,
        future_dim: int = 256,
        action_horizon: int = 8,
        action_dim: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        input_dim = int(decision_hidden_dim) + int(future_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
        )

    def forward(self, decision_hidden: torch.Tensor, future_slots: torch.Tensor) -> torch.Tensor:
        if decision_hidden.dim() == 3:
            d = decision_hidden.mean(dim=1)
        else:
            d = decision_hidden.flatten(start_dim=1)
        e = future_slots.mean(dim=1)
        y = self.net(torch.cat([d, e], dim=-1))
        return y.view(future_slots.shape[0], self.action_horizon, self.action_dim)


@dataclass
class ThreeStageDistillConfig:
    use_three_stage_distill: bool = False
    stage1_epochs: int = 2
    stage1_iters: int = -1
    stage2_end_epochs: int = 7
    lambda_init: float = 1.0
    lambda_corr: float = 1.0
    lambda_cons: float = 0.1
    lambda_teacher_min: float = 0.0
    teacher_decay_type: str = "linear"
    beta_cos: float = 0.5
    use_soft_intervention: bool = True
    use_uncertainty: bool = False
    use_self_consistency: bool = False
    use_counterfactual_gain: bool = False
    use_failure_type_attribution: bool = False
    tau_f: float = 0.5
    tau_d: float = 0.5
    tau_u: float = 0.1
    soft_a: float = 1.0
    soft_b: float = 1.0
    soft_c: float = 1.0
    soft_d: float = 1.0
    soft_threshold: float = 0.0
    alpha_l1: float = 1.0
    alpha_collision: float = 1.0
    alpha_offroad: float = 1.0
    alpha_route: float = 1.0
    alpha_ttc: float = 1.0
    eps: float = 1e-6
    debug_sanity: bool = True
    adaptive_stage_enabled: bool = False
    adaptive_stage1_min_epochs: int = 0
    adaptive_stage2_min_epochs: int = 0
    adaptive_stage1_patience: int = 200
    adaptive_stage2_patience: int = 200
    adaptive_stage1_delta: float = 1e-4
    adaptive_stage1_ema_decay: float = 0.95
    adaptive_stage2_ema_decay: float = 0.95
    adaptive_stage2_gate_threshold: float = 0.35
    adaptive_stage2_intervention_threshold: float = 0.20
    adaptive_teacher_anneal_epochs: int = 20
    use_decision_efficiency: bool = True
    decision_efficiency_stage: str = "stage2"
    decision_efficiency_start_epoch: Optional[int] = None
    decision_efficiency_tau: float = 0.5
    decision_efficiency_eps: float = 0.0
    decision_efficiency_weight_min: float = 0.05
    decision_efficiency_weight_max: float = 1.0
    decision_efficiency_normalize: bool = True
    decision_efficiency_slot_level: bool = True
    decision_efficiency_mask_type: str = "learnable_null"
    decision_efficiency_distance: str = "traj_l1"
    decision_efficiency_top_pool: str = "softmax"
    lambda_aux_plan: float = 0.05
    lambda_decision_efficiency_distill: float = 1.0
    detach_decision_hidden_for_aux: bool = True
    enable_decision_efficiency_stage1: bool = False
    enable_decision_efficiency_stage2: bool = True
    enable_decision_efficiency_stage3: bool = False
    stage1_aux_warmup: bool = True
    decision_efficiency_slot_dim: int = 256
    decision_hidden_dim: int = 1536
    decision_aux_hidden_dim: int = 256



def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    try:
        return torch.as_tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError):
        return None


class ThreeStageImaginationDistiller(nn.Module):
    """Plugin-style controller for LatentSight three-stage imagination distillation.

    The module is intentionally independent from the planner forward. It consumes
    tensors already produced by the student, teacher projector, and optional debug
    forwards, then returns additive losses and scalar logs.
    """

    def __init__(self, cfg: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        merged: Dict[str, Any] = {}
        for field_name in ThreeStageDistillConfig.__dataclass_fields__:
            merged[field_name] = _cfg_get(cfg, field_name, getattr(ThreeStageDistillConfig(), field_name))
        merged.update(kwargs)
        self.cfg = ThreeStageDistillConfig(**merged)
        self._printed_sanity = False

        self.aux_traj_head = AuxiliaryPlanningProbe(
            decision_hidden_dim=int(self.cfg.decision_hidden_dim),
            future_dim=int(self.cfg.decision_efficiency_slot_dim),
            hidden_dim=int(self.cfg.decision_aux_hidden_dim),
        )
        self.decision_null_slot = nn.Parameter(torch.zeros(1, 1, int(self.cfg.decision_efficiency_slot_dim)))

        self.register_buffer("adaptive_stage_idx", torch.tensor(1.0), persistent=True)
        self.register_buffer("adaptive_stage1_loss_ema", torch.tensor(float("inf")), persistent=True)
        self.register_buffer("adaptive_stage1_best", torch.tensor(float("inf")), persistent=True)
        self.register_buffer("adaptive_stage1_bad_steps", torch.tensor(0.0), persistent=True)
        self.register_buffer("adaptive_stage2_gate_ema", torch.tensor(1.0), persistent=True)
        self.register_buffer("adaptive_stage2_intervention_ema", torch.tensor(1.0), persistent=True)
        self.register_buffer("adaptive_stage2_low_steps", torch.tensor(0.0), persistent=True)
        self.register_buffer("adaptive_stage2_start_epoch", torch.tensor(-1.0), persistent=True)
        self.register_buffer("adaptive_stage3_start_epoch", torch.tensor(-1.0), persistent=True)

    def current_stage(self, epoch: int = 0, iteration: int = 0) -> str:
        if self.cfg.adaptive_stage_enabled:
            idx = int(self.adaptive_stage_idx.detach().item())
            if idx <= 1:
                return "init"
            if idx == 2:
                return "corrective"
            return "self_evolving"
        if self.cfg.stage1_iters is not None and self.cfg.stage1_iters >= 0 and iteration < self.cfg.stage1_iters:
            return "init"
        if epoch < self.cfg.stage1_epochs:
            return "init"
        if epoch < self.cfg.stage2_end_epochs:
            return "corrective"
        return "self_evolving"

    def teacher_lambda(self, epoch: int = 0) -> float:
        if self.cfg.adaptive_stage_enabled:
            stage = self.current_stage(epoch)
            if stage != "self_evolving":
                return float(self.cfg.lambda_corr)
            start_epoch = int(self.adaptive_stage3_start_epoch.detach().item())
            if start_epoch < 0:
                start_epoch = epoch
            span = max(1, int(self.cfg.adaptive_teacher_anneal_epochs))
            progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / span))
        else:
            if epoch < self.cfg.stage2_end_epochs:
                return float(self.cfg.lambda_corr)
            span = max(1, self.cfg.stage2_end_epochs - self.cfg.stage1_epochs)
            progress = min(1.0, max(0.0, (epoch - self.cfg.stage2_end_epochs + 1) / span))
        if self.cfg.teacher_decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0 - progress
        return float(self.cfg.lambda_teacher_min + (self.cfg.lambda_corr - self.cfg.lambda_teacher_min) * factor)

    @torch.no_grad()
    def update_adaptive_stage(
        self,
        *,
        stage: str,
        epoch: int,
        iteration: int,
        hidden_loss: torch.Tensor,
        teacher_gate_mean: torch.Tensor,
        intervention_weight_mean: torch.Tensor,
    ) -> None:
        if not self.cfg.adaptive_stage_enabled or not self.training:
            return
        if stage == "init":
            decay = float(self.cfg.adaptive_stage1_ema_decay)
            value = hidden_loss.detach().float().mean()
            if not torch.isfinite(self.adaptive_stage1_loss_ema):
                self.adaptive_stage1_loss_ema.copy_(value)
            else:
                self.adaptive_stage1_loss_ema.mul_(decay).add_(value * (1.0 - decay))
            if self.adaptive_stage1_loss_ema < self.adaptive_stage1_best - float(self.cfg.adaptive_stage1_delta):
                self.adaptive_stage1_best.copy_(self.adaptive_stage1_loss_ema)
                self.adaptive_stage1_bad_steps.zero_()
            else:
                self.adaptive_stage1_bad_steps.add_(1.0)
            min_epoch_ready = epoch >= max(int(self.cfg.stage1_epochs), int(self.cfg.adaptive_stage1_min_epochs))
            iter_ready = self.cfg.stage1_iters is None or self.cfg.stage1_iters < 0 or iteration >= int(self.cfg.stage1_iters)
            if min_epoch_ready and iter_ready and self.adaptive_stage1_bad_steps >= float(self.cfg.adaptive_stage1_patience):
                self.adaptive_stage_idx.fill_(2.0)
                if self.adaptive_stage2_start_epoch.item() < 0:
                    self.adaptive_stage2_start_epoch.fill_(float(epoch))
        elif stage == "corrective":
            decay = float(self.cfg.adaptive_stage2_ema_decay)
            gate = teacher_gate_mean.detach().float().mean().clamp(0.0, 1.0)
            weight = intervention_weight_mean.detach().float().mean().clamp(0.0, 1.0)
            self.adaptive_stage2_gate_ema.mul_(decay).add_(gate * (1.0 - decay))
            self.adaptive_stage2_intervention_ema.mul_(decay).add_(weight * (1.0 - decay))
            low_gate = self.adaptive_stage2_gate_ema <= float(self.cfg.adaptive_stage2_gate_threshold)
            low_weight = self.adaptive_stage2_intervention_ema <= float(self.cfg.adaptive_stage2_intervention_threshold)
            if bool(low_gate.item() or low_weight.item()):
                self.adaptive_stage2_low_steps.add_(1.0)
            else:
                self.adaptive_stage2_low_steps.zero_()
            start_epoch = int(self.adaptive_stage2_start_epoch.item())
            if start_epoch < 0:
                start_epoch = int(self.cfg.stage1_epochs)
            min_epoch_ready = (epoch - start_epoch) >= int(self.cfg.adaptive_stage2_min_epochs)
            if min_epoch_ready and self.adaptive_stage2_low_steps >= float(self.cfg.adaptive_stage2_patience):
                self.adaptive_stage_idx.fill_(3.0)
                if self.adaptive_stage3_start_epoch.item() < 0:
                    self.adaptive_stage3_start_epoch.fill_(float(epoch))

    def align_tokens(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[1] != target.shape[1]:
            source = F.interpolate(source.transpose(1, 2), size=target.shape[1], mode="nearest").transpose(1, 2)
        if source.shape[-1] != target.shape[-1]:
            if source.shape[-1] > target.shape[-1]:
                source = source[..., : target.shape[-1]]
            else:
                pad = target.shape[-1] - source.shape[-1]
                source = F.pad(source, (0, pad))
        return source

    def dist_per_sample(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
        z_teacher = self.align_tokens(z_teacher, h_future)
        h_norm = F.normalize(h_future.float(), dim=-1)
        z_norm = F.normalize(z_teacher.float(), dim=-1)
        mse = (h_norm - z_norm).pow(2).mean(dim=(1, 2))
        cos = 1.0 - F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1).mean(dim=1)
        return mse + self.cfg.beta_cos * cos

    def dist_mean(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_teacher = self.align_tokens(z_teacher, h_future)
        h_norm = F.normalize(h_future.float(), dim=-1)
        z_norm = F.normalize(z_teacher.float(), dim=-1)
        mse = F.mse_loss(h_norm, z_norm, reduction="mean")
        cos_sim = F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        loss = mse + self.cfg.beta_cos * cos_loss
        logs = {
            "loss_hidden_mse": mse.detach(),
            "loss_hidden_cos": cos_loss.detach(),
            "hidden_cos_sim": cos_sim.detach(),
            "H_future_norm": h_future.detach().float().norm(dim=-1).mean(),
            "Z_teacher_norm": z_teacher.detach().float().norm(dim=-1).mean(),
        }
        return loss, logs

    def _per_sample_l1(self, pred: Optional[torch.Tensor], gt: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if pred is None or gt is None:
            return torch.zeros(batch_size, device=device)
        gt = gt.to(device=pred.device, dtype=pred.dtype)
        if pred.shape != gt.shape:
            min_dims = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
            pred = pred[tuple(slice(0, d) for d in min_dims)]
            gt = gt[tuple(slice(0, d) for d in min_dims)]
        return (pred.float() - gt.float()).abs().flatten(start_dim=1).mean(dim=1)

    def failure_score(
        self,
        traj_pred: Optional[torch.Tensor],
        traj_gt: Optional[torch.Tensor],
        extras: Optional[Mapping[str, Any]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        score = self.cfg.alpha_l1 * self._per_sample_l1(traj_pred, traj_gt, batch_size, device)
        extras = extras or {}
        optional_terms = [
            ("collision_risk", self.cfg.alpha_collision),
            ("offroad_risk", self.cfg.alpha_offroad),
            ("route_deviation", self.cfg.alpha_route),
            ("ttc_violation", self.cfg.alpha_ttc),
        ]
        for key, alpha in optional_terms:
            value = _as_tensor(extras.get(key), device, score.dtype)
            if value is not None:
                score = score + alpha * value.flatten(start_dim=1).mean(dim=1) if value.ndim > 1 else score + alpha * value.flatten()
        return score

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.numel() <= 1:
            return torch.zeros_like(x)
        std = x.std(unbiased=False)
        if not torch.isfinite(std) or std < self.cfg.eps:
            return torch.zeros_like(x)
        return (x - x.mean()) / (std + self.cfg.eps)

    def intervention_weight(self, failure: torch.Tensor, discrepancy: torch.Tensor, uncertainty: torch.Tensor, gain: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.use_soft_intervention:
            logits = (
                self.cfg.soft_a * self._batch_norm(failure)
                + self.cfg.soft_b * self._batch_norm(discrepancy)
                + self.cfg.soft_c * self._batch_norm(uncertainty)
                + self.cfg.soft_d * self._batch_norm(gain)
                - self.cfg.soft_threshold
            )
            weights = torch.sigmoid(logits)
            selected = (weights > 0.5).float().mean()
            logs = {"selected_sample_ratio": selected.detach(), "intervention_ratio": selected.detach()}
        else:
            mask = (failure > self.cfg.tau_f) & ((discrepancy > self.cfg.tau_d) | (uncertainty > self.cfg.tau_u))
            weights = mask.to(dtype=failure.dtype)
            logs = {"selected_sample_ratio": weights.mean().detach(), "intervention_ratio": weights.mean().detach()}
        return weights, logs

    def weighted_distill(self, h_future: torch.Tensor, z_teacher: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        per_sample = self.dist_per_sample(h_future, z_teacher)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights)).to(per_sample.dtype)
        denom = weights.sum().clamp_min(self.cfg.eps)
        if weights.sum().detach() <= self.cfg.eps:
            return per_sample.sum() * 0.0
        return (per_sample * weights).sum() / denom
    
    def _per_sample_traj_l1(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = gt.to(device=pred.device, dtype=pred.dtype)
        if pred.shape != gt.shape:
            min_dims = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
            pred = pred[tuple(slice(0, d) for d in min_dims)]
            gt = gt[tuple(slice(0, d) for d in min_dims)]
        return (pred.float() - gt.float()).abs().flatten(start_dim=1).mean(dim=1)

    def _decision_efficiency_enabled(self, stage: str, epoch: int) -> bool:
        if not self.cfg.use_decision_efficiency:
            return False
        start_epoch = self.cfg.decision_efficiency_start_epoch
        if start_epoch is not None and epoch < int(start_epoch):
            return False
        if stage == "init":
            return bool(self.cfg.enable_decision_efficiency_stage1)
        if stage == "corrective":
            return bool(self.cfg.enable_decision_efficiency_stage2)
        return bool(self.cfg.enable_decision_efficiency_stage3)

    def _replacement_slot(self, future_slots: torch.Tensor) -> torch.Tensor:
        mask_type = self.cfg.decision_efficiency_mask_type
        if mask_type == "zero":
            return torch.zeros(future_slots.shape[0], future_slots.shape[-1], device=future_slots.device, dtype=future_slots.dtype)
        if mask_type == "batch_mean":
            return future_slots.detach().mean(dim=(0, 1), keepdim=False).unsqueeze(0).expand(future_slots.shape[0], -1).to(dtype=future_slots.dtype)
        if mask_type == "learnable_null":
            if self.decision_null_slot.shape[-1] == future_slots.shape[-1]:
                return self.decision_null_slot.to(device=future_slots.device, dtype=future_slots.dtype).expand(future_slots.shape[0], 1, -1).squeeze(1)
            return torch.zeros(future_slots.shape[0], future_slots.shape[-1], device=future_slots.device, dtype=future_slots.dtype)
        raise ValueError(f"Unsupported decision_efficiency_mask_type={mask_type}")

    def dist_per_slot(self, h_future: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
        z_teacher = self.align_tokens(z_teacher, h_future)
        h_norm = F.normalize(h_future.float(), dim=-1)
        z_norm = F.normalize(z_teacher.float(), dim=-1)
        mse = (h_norm - z_norm).pow(2).mean(dim=2)
        cos = 1.0 - F.cosine_similarity(h_future.float(), z_teacher.float(), dim=-1)
        return mse + self.cfg.beta_cos * cos

    def decision_efficiency(
        self,
        *,
        decision_hidden: torch.Tensor,
        h_future: torch.Tensor,
        traj_pred: torch.Tensor,
        traj_gt: torch.Tensor,
        base_weight: torch.Tensor,
        stage: str,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert h_future.dim() == 3, "H_future must be [B,K,C] for slot-level decision efficiency"
        B, K, _ = h_future.shape
        d_aux = decision_hidden.detach() if self.cfg.detach_decision_hidden_for_aux else decision_hidden
        y_aux_full = self.aux_traj_head(d_aux, h_future)
        assert y_aux_full.shape == traj_pred.shape, "Y_aux_full must match Y_base/traj_pred shape"
        l_base_each = self._per_sample_traj_l1(traj_pred.detach(), traj_gt)
        l_aux_each = self._per_sample_traj_l1(y_aux_full, traj_gt)
        assert l_base_each.shape == (B,)
        assert l_aux_each.shape == (B,)
        enabled = self._decision_efficiency_enabled(stage, epoch)
        if enabled:
            replacement = self._replacement_slot(h_future)
            s_list = []
            for k in range(K):
                e_mask = h_future.clone()
                e_mask[:, k, :] = replacement
                y_mask = self.aux_traj_head(d_aux, e_mask)
                s_list.append((y_aux_full.detach() - y_mask.detach()).abs().flatten(start_dim=1).mean(dim=1))
            S = torch.stack(s_list, dim=1)
            delta_l = l_base_each.detach() - l_aux_each.detach()
            if self.cfg.decision_efficiency_normalize:
                s_std = S.std(dim=1, keepdim=True, unbiased=False)
                S_score = (S - S.mean(dim=1, keepdim=True)) / (s_std + 1e-6)
                d_std = delta_l.std(unbiased=False)
                delta_score = (delta_l - delta_l.mean()) / (d_std + 1e-6)
            else:
                S_score, delta_score = S, delta_l
            score = S_score + delta_score[:, None]
            decision_weight_k = torch.sigmoid((score - float(self.cfg.decision_efficiency_eps)) / max(float(self.cfg.decision_efficiency_tau), 1e-6))
            decision_weight_k = decision_weight_k.clamp(float(self.cfg.decision_efficiency_weight_min), float(self.cfg.decision_efficiency_weight_max)).detach()
        else:
            S = torch.zeros(B, K, device=h_future.device, dtype=h_future.dtype)
            delta_l = l_base_each.detach() - l_aux_each.detach()
            decision_weight_k = torch.ones(B, K, device=h_future.device, dtype=h_future.dtype)
        final_weight_k = (base_weight.detach()[:, None] * decision_weight_k).detach()
        assert decision_weight_k.shape == (B, K)
        assert final_weight_k.shape == (B, K)
        logs = {
            "decision_efficiency_enabled": torch.tensor(1.0 if enabled else 0.0, device=h_future.device),
            "decision_L_base_mean": l_base_each.detach().mean(),
            "decision_L_aux_mean": l_aux_each.detach().mean(),
            "decision_delta_L_mean": delta_l.detach().mean(),
            "decision_delta_L_std": delta_l.detach().std(unbiased=False),
            "decision_positive_ratio": (delta_l.detach() > 0).float().mean(),
            "decision_S_mean": S.detach().mean(),
            "decision_S_std": S.detach().std(unbiased=False),
            "decision_S_max": S.detach().max() if S.numel() else torch.tensor(0.0, device=h_future.device),
            "decision_S_min": S.detach().min() if S.numel() else torch.tensor(0.0, device=h_future.device),
            "decision_weight_mean": decision_weight_k.mean(),
            "decision_weight_max": decision_weight_k.max(),
            "decision_weight_min": decision_weight_k.min(),
            "base_weight_mean": base_weight.detach().mean(),
            "final_weight_k_mean": final_weight_k.mean(),
            "lambda_aux_plan": torch.tensor(float(self.cfg.lambda_aux_plan), device=h_future.device),
        }
        return final_weight_k, l_aux_each.mean(), logs

    def weighted_distill_slots(self, h_future: torch.Tensor, z_teacher: torch.Tensor, weights_k: torch.Tensor) -> torch.Tensor:
        per_slot = self.dist_per_slot(h_future, z_teacher)
        weights_k = torch.where(torch.isfinite(weights_k), weights_k, torch.zeros_like(weights_k)).to(per_slot.dtype)
        if weights_k.sum().detach() <= self.cfg.eps:
            return per_slot.sum() * 0.0
        return (per_slot * weights_k).sum(dim=1).mean()

    def compute_consistency_loss(
        self,
        h_future: torch.Tensor,
        h_future_ref: Optional[torch.Tensor],
        mask_or_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h_future_ref is None:
            return h_future.sum() * 0.0
        h_future_ref = self.align_tokens(h_future_ref.detach(), h_future)
        per_sample = self.dist_per_sample(h_future, h_future_ref)
        if mask_or_weight is None:
            return per_sample.mean()
        weight = mask_or_weight.to(device=per_sample.device, dtype=per_sample.dtype)
        if weight.sum().detach() <= self.cfg.eps:
            return per_sample.sum() * 0.0
        return (per_sample * weight).sum() / weight.sum().clamp_min(self.cfg.eps)

    def sanity_check(self, tensors: Mapping[str, Optional[torch.Tensor]], logs: Mapping[str, torch.Tensor], stage: str) -> None:
        if self._printed_sanity or not self.cfg.debug_sanity:
            return
        self._printed_sanity = True
        h = tensors.get("H_future")
        z = tensors.get("Z_teacher")
        pred = tensors.get("traj_pred")
        gt = tensors.get("traj_gt")
        def _shape(x: Optional[torch.Tensor]) -> str:
            return "None" if x is None else str(tuple(x.shape))
        def _stats(name: str, x: Optional[torch.Tensor]) -> str:
            if x is None:
                return f"{name}=None"
            xf = x.detach().float()
            return f"{name}: mean={xf.mean().item():.6f} std={xf.std(unbiased=False).item():.6f} norm={xf.norm(dim=-1).mean().item():.6f}"
        hidden_finite = bool(torch.isfinite(logs.get("loss_hidden_init", torch.tensor(0.0, device=h.device if h is not None else "cpu"))).all().item())
        weight = logs.get("intervention_weight_mean")
        weight_ok = weight is None or bool(torch.isfinite(weight).all().item())
        print(
            "[LatentSight][ThreeStage][Sanity] "
            f"stage={stage} H_future_shape={_shape(h)} Z_teacher_shape={_shape(z)} "
            f"traj_pred_shape={_shape(pred)} traj_gt_shape={_shape(gt)} "
            f"{_stats('H_future', h)} {_stats('Z_teacher', z)} "
            f"hidden_loss_finite={hidden_finite} intervention_weight_ok={weight_ok}"
        )

    def forward(
        self,
        *,
        h_future: Optional[torch.Tensor],
        z_teacher: Optional[torch.Tensor],
        traj_pred: Optional[torch.Tensor] = None,
        traj_gt: Optional[torch.Tensor] = None,
        h_future_aug: Optional[torch.Tensor] = None,
        h_future_ref: Optional[torch.Tensor] = None,
        extras: Optional[Mapping[str, Any]] = None,
        epoch: int = 0,
        iteration: int = 0,
        teacher_enabled: bool = True,
        teacher_requires_grad_count: int = 0,
        teacher_gate: Optional[torch.Tensor] = None,
        decision_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = h_future.device if h_future is not None else (z_teacher.device if z_teacher is not None else torch.device("cpu"))
        zero = torch.tensor(0.0, device=device)
        stage = self.current_stage(epoch, iteration)
        logs: Dict[str, torch.Tensor] = {
            "loss_hidden_init": zero,
            "loss_corrective": zero,
            "loss_consistency": zero,
            "lambda_teacher": torch.tensor(self.teacher_lambda(epoch), device=device),
            "current_distill_stage": torch.tensor({"init": 1.0, "corrective": 2.0, "self_evolving": 3.0}[stage], device=device),
            "current_distill_stage_init": torch.tensor(1.0 if stage == "init" else 0.0, device=device),
            "current_distill_stage_corrective": torch.tensor(1.0 if stage == "corrective" else 0.0, device=device),
            "current_distill_stage_self_evolving": torch.tensor(1.0 if stage == "self_evolving" else 0.0, device=device),
            "use_three_stage_distill": torch.tensor(1.0 if self.cfg.use_three_stage_distill else 0.0, device=device),
            "use_counterfactual_gain": torch.tensor(1.0 if self.cfg.use_counterfactual_gain else 0.0, device=device),
            "use_uncertainty": torch.tensor(1.0 if self.cfg.use_uncertainty else 0.0, device=device),
            "use_self_consistency": torch.tensor(1.0 if self.cfg.use_self_consistency else 0.0, device=device),
            "teacher_enabled": torch.tensor(1.0 if teacher_enabled else 0.0, device=device),
            "teacher_requires_grad_count": torch.tensor(float(teacher_requires_grad_count), device=device),
            "teacher_gate_mean": zero,
            "failure_score_mean": zero,
            "discrepancy_score_mean": zero,
            "uncertainty_score_mean": zero,
            "intervention_weight_mean": zero,
            "intervention_weight_max": zero,
            "intervention_ratio": zero,
            "selected_sample_ratio": zero,
            "hidden_cos_sim": zero,
            "H_future_norm": zero,
            "Z_teacher_norm": zero,
            "adaptive_stage_enabled": torch.tensor(1.0 if self.cfg.adaptive_stage_enabled else 0.0, device=device),
            "adaptive_stage_idx": self.adaptive_stage_idx.detach().to(device=device),
            "adaptive_stage1_loss_ema": self.adaptive_stage1_loss_ema.detach().to(device=device),
            "adaptive_stage1_bad_steps": self.adaptive_stage1_bad_steps.detach().to(device=device),
            "adaptive_stage2_gate_ema": self.adaptive_stage2_gate_ema.detach().to(device=device),
            "adaptive_stage2_intervention_ema": self.adaptive_stage2_intervention_ema.detach().to(device=device),
            "adaptive_stage2_low_steps": self.adaptive_stage2_low_steps.detach().to(device=device),
            "adaptive_stage3_start_epoch": self.adaptive_stage3_start_epoch.detach().to(device=device),
        }
        #losses = {"loss_hidden": zero, "loss_corrective": zero, "loss_consistency": zero}
        losses = {"loss_hidden": zero, "loss_corrective": zero, "loss_consistency": zero, "loss_aux_plan": zero}

        if not self.cfg.use_three_stage_distill or not teacher_enabled or h_future is None or z_teacher is None:
            return losses, logs

        z_teacher = self.align_tokens(z_teacher.to(device=h_future.device, dtype=h_future.dtype), h_future)
        hidden_loss, hidden_logs = self.dist_mean(h_future, z_teacher)
        logs.update(hidden_logs)
        logs["loss_hidden_init"] = hidden_loss.detach()
        discrepancy = self.dist_per_sample(h_future.detach(), z_teacher.detach())
        batch_size = h_future.shape[0]

        if stage == "init":
            losses["loss_hidden"] = self.cfg.lambda_init * hidden_loss
        else:
            failure = self.failure_score(traj_pred, traj_gt, extras, batch_size, h_future.device)
            if self.cfg.use_uncertainty and h_future_aug is not None:
                uncertainty = self.dist_per_sample(h_future.detach(), h_future_aug.detach())
            else:
                uncertainty = torch.zeros_like(failure)
            gain = torch.zeros_like(failure)
            weights, weight_logs = self.intervention_weight(failure, discrepancy, uncertainty, gain)
            if teacher_gate is not None:
                gate = teacher_gate.detach().to(device=weights.device, dtype=weights.dtype)
                if gate.ndim == 0:
                    gate = gate.expand_as(weights)
                else:
                    gate = gate.flatten()
                    if gate.numel() == 1:
                        gate = gate.expand_as(weights)
                    elif gate.numel() != weights.numel():
                        gate = gate[: weights.numel()] if gate.numel() > weights.numel() else F.pad(gate, (0, weights.numel() - gate.numel()))
                weights = weights * gate.clamp(0.0, 1.0)
            corr = self.weighted_distill(h_future, z_teacher, weights)
            base_weight = weights
            final_weight_k = None
            aux_loss = h_future.sum() * 0.0
            if self._decision_efficiency_enabled(stage, epoch) and decision_hidden is not None and traj_pred is not None and traj_gt is not None and self.cfg.decision_efficiency_slot_level:
                final_weight_k, aux_loss, de_logs = self.decision_efficiency(
                    decision_hidden=decision_hidden,
                    h_future=h_future,
                    traj_pred=traj_pred.to(device=h_future.device, dtype=h_future.dtype),
                    traj_gt=traj_gt.to(device=h_future.device, dtype=h_future.dtype),
                    base_weight=base_weight,
                    stage=stage,
                    epoch=epoch,
                )
                logs.update(de_logs)
                corr = self.weighted_distill_slots(h_future, z_teacher, final_weight_k)
                losses["loss_aux_plan"] = float(self.cfg.lambda_aux_plan) * aux_loss
            else:
                corr = self.weighted_distill(h_future, z_teacher, weights)
            logs.update(weight_logs)
            logs["teacher_gate_mean"] = (teacher_gate.detach().float().mean().to(device) if teacher_gate is not None else torch.tensor(1.0, device=device))
            logs["failure_score_mean"] = failure.detach().mean()
            logs["discrepancy_score_mean"] = discrepancy.detach().mean()
            logs["uncertainty_score_mean"] = uncertainty.detach().mean()
            logs["intervention_weight_mean"] = weights.detach().mean()
            logs["intervention_weight_max"] = weights.detach().max() if weights.numel() else zero
            if stage == "corrective":
                #losses["loss_corrective"] = self.cfg.lambda_corr * corr
                losses["loss_corrective"] = self.cfg.lambda_corr * float(self.cfg.lambda_decision_efficiency_distill) * corr
            else:
                teacher_weight = self.teacher_lambda(epoch)
                losses["loss_corrective"] = teacher_weight * corr
                low_failure_weight = (1.0 - weights).detach()
                ref = h_future_ref if h_future_ref is not None else (h_future_aug if self.cfg.use_uncertainty else None)
                cons = self.compute_consistency_loss(h_future, ref, low_failure_weight) if self.cfg.use_self_consistency else h_future.sum() * 0.0
                losses["loss_consistency"] = self.cfg.lambda_cons * cons
                logs["loss_consistency"] = cons.detach()
            logs["loss_corrective"] = corr.detach()
            logs["raw_distill_loss"] = self.dist_per_sample(h_future.detach(), z_teacher.detach()).mean().detach()
            logs["weighted_corr_distill_loss"] = corr.detach()
            logs["loss_aux_plan"] = losses["loss_aux_plan"].detach()

        self.update_adaptive_stage(
            stage=stage,
            epoch=epoch,
            iteration=iteration,
            hidden_loss=logs["loss_hidden_init"],
            teacher_gate_mean=logs["teacher_gate_mean"],
            intervention_weight_mean=logs["intervention_weight_mean"],
        )
        logs["adaptive_stage_idx"] = self.adaptive_stage_idx.detach().to(device=device)
        logs["adaptive_stage1_loss_ema"] = self.adaptive_stage1_loss_ema.detach().to(device=device)
        logs["adaptive_stage1_bad_steps"] = self.adaptive_stage1_bad_steps.detach().to(device=device)
        logs["adaptive_stage2_gate_ema"] = self.adaptive_stage2_gate_ema.detach().to(device=device)
        logs["adaptive_stage2_intervention_ema"] = self.adaptive_stage2_intervention_ema.detach().to(device=device)
        logs["adaptive_stage2_low_steps"] = self.adaptive_stage2_low_steps.detach().to(device=device)
        logs["adaptive_stage3_start_epoch"] = self.adaptive_stage3_start_epoch.detach().to(device=device)
        logs["lambda_teacher"] = torch.tensor(self.teacher_lambda(epoch), device=device)

        self.sanity_check({"H_future": h_future, "Z_teacher": z_teacher, "traj_pred": traj_pred, "traj_gt": traj_gt}, logs, stage)
        return losses, logs


def build_imagination_distiller(cfg: Optional[Any] = None, **kwargs: Any) -> ThreeStageImaginationDistiller:
    return ThreeStageImaginationDistiller(cfg, **kwargs)
