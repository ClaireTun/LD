from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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

    def current_stage(self, epoch: int = 0, iteration: int = 0) -> str:
        if self.cfg.stage1_iters is not None and self.cfg.stage1_iters >= 0 and iteration < self.cfg.stage1_iters:
            return "init"
        if epoch < self.cfg.stage1_epochs:
            return "init"
        if epoch < self.cfg.stage2_end_epochs:
            return "corrective"
        return "self_evolving"

    def teacher_lambda(self, epoch: int = 0) -> float:
        if epoch < self.cfg.stage2_end_epochs:
            return float(self.cfg.lambda_corr)
        span = max(1, self.cfg.stage2_end_epochs - self.cfg.stage1_epochs)
        progress = min(1.0, max(0.0, (epoch - self.cfg.stage2_end_epochs + 1) / span))
        if self.cfg.teacher_decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0 - progress
        return float(self.cfg.lambda_teacher_min + (self.cfg.lambda_corr - self.cfg.lambda_teacher_min) * factor)

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
        }
        losses = {"loss_hidden": zero, "loss_corrective": zero, "loss_consistency": zero}

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
            corr = self.weighted_distill(h_future, z_teacher, weights)
            logs.update(weight_logs)
            logs["failure_score_mean"] = failure.detach().mean()
            logs["discrepancy_score_mean"] = discrepancy.detach().mean()
            logs["uncertainty_score_mean"] = uncertainty.detach().mean()
            logs["intervention_weight_mean"] = weights.detach().mean()
            logs["intervention_weight_max"] = weights.detach().max() if weights.numel() else zero
            if stage == "corrective":
                losses["loss_corrective"] = self.cfg.lambda_corr * corr
            else:
                teacher_weight = self.teacher_lambda(epoch)
                losses["loss_corrective"] = teacher_weight * corr
                low_failure_weight = (1.0 - weights).detach()
                ref = h_future_ref if h_future_ref is not None else (h_future_aug if self.cfg.use_uncertainty else None)
                cons = self.compute_consistency_loss(h_future, ref, low_failure_weight) if self.cfg.use_self_consistency else h_future.sum() * 0.0
                losses["loss_consistency"] = self.cfg.lambda_cons * cons
                logs["loss_consistency"] = cons.detach()
            logs["loss_corrective"] = corr.detach()

        self.sanity_check({"H_future": h_future, "Z_teacher": z_teacher, "traj_pred": traj_pred, "traj_gt": traj_gt}, logs, stage)
        return losses, logs


def build_imagination_distiller(cfg: Optional[Any] = None, **kwargs: Any) -> ThreeStageImaginationDistiller:
    return ThreeStageImaginationDistiller(cfg, **kwargs)
