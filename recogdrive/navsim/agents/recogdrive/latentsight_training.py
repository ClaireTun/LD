# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# import torch
# from torch import nn


# LORA_TARGET_SUFFIXES = (
#     "q_proj",
#     "k_proj",
#     "v_proj",
#     "o_proj",
#     "gate_proj",
#     "up_proj",
#     "down_proj",
# )


# class LoRALinear(nn.Module):
#     """Minimal LoRA wrapper for an existing nn.Linear layer.

#     The wrapped base layer remains frozen; only lora_A/lora_B are trainable.
#     """

#     def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
#         super().__init__()
#         if r <= 0:
#             raise ValueError(f"LoRA rank must be positive, got {r}")
#         self.base = base
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.lora_A = nn.Linear(base.in_features, r, bias=False)
#         self.lora_B = nn.Linear(r, base.out_features, bias=False)
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
#         nn.init.zeros_(self.lora_B.weight)

#         # The wrapped VLM layer is often already on CUDA and may use bf16.
#         # Newly-created LoRA modules default to CPU/fp32, so place them beside
#         # the base layer immediately. Keep LoRA weights in fp32 for stable
#         # optimization; forward casts the activation only for the LoRA branch.
#         base_weight = self.base.weight
#         self.lora_A.to(device=base_weight.device, dtype=torch.float32)
#         self.lora_B.to(device=base_weight.device, dtype=torch.float32)
#         for p in self.base.parameters():
#             p.requires_grad = False

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         base_out = self.base(x)
#         lora_in = self.dropout(x).to(device=self.lora_A.weight.device, dtype=self.lora_A.weight.dtype)
#         lora_out = self.lora_B(self.lora_A(lora_in)) * self.scaling
#         return base_out + lora_out.to(dtype=base_out.dtype, device=base_out.device)


# def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
#     parts = module_name.split(".")
#     parent = root
#     for part in parts[:-1]:
#         parent = getattr(parent, part)
#     return parent, parts[-1]


# def inject_lora(
#     module: Optional[nn.Module],
#     target_keywords: Optional[Sequence[str]] = None,
#     r: int = 8,
#     alpha: int = 16,
#     dropout: float = 0.05,
# ) -> List[str]:
#     if module is None:
#         return []
#     target_keywords = tuple(target_keywords or LORA_TARGET_SUFFIXES)
#     injected: List[str] = []
#     for name, child in list(module.named_modules()):
#         if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
#             continue
#         if not any(name.endswith(k) or k in name for k in target_keywords):
#             continue
#         parent, attr = _get_parent_module(module, name)
#         setattr(parent, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
#         injected.append(name)
#     print(f"[LatentSight][LoRA] injected {len(injected)} Linear layers; sample={injected[:20]}")
#     return injected


# def freeze_teacher(model: nn.Module) -> None:
#     teacher = getattr(model, "sgdrive_teacher", None)
#     if teacher is None:
#         return
#     teacher.eval()
#     for p in teacher.parameters():
#         p.requires_grad = False
#     teacher_agent = getattr(teacher, "teacher_agent", None)
#     if teacher_agent is not None:
#         teacher_agent.eval()
#         for p in teacher_agent.parameters():
#             p.requires_grad = False


# def _set_named_modules_trainable(model: nn.Module, keywords: Sequence[str], trainable: bool = True) -> None:
#     for name, p in model.named_parameters():
#         if any(k in name for k in keywords):
#             p.requires_grad = trainable


# def _print_module_grad_state(model: nn.Module) -> None:
#     rows = []
#     for name, module in model.named_children():
#         total = sum(p.numel() for p in module.parameters())
#         trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
#         if total:
#             rows.append((name, trainable, total, trainable / total))
#     for name, trainable, total, ratio in rows:
#         print(f"[LatentSight][Trainable] module={name:<32} trainable={trainable:,}/{total:,} ({ratio:.4%})")


# def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return trainable, total, trainable / max(total, 1)


# def set_latentsight_trainable(model: nn.Module, mode: str, cfg: Optional[Any] = None) -> Tuple[int, int, float]:
#     """Central trainable-module switch for LatentSight experiments."""

#     mode = (mode or "legacy").replace("-", "_").lower()
#     if mode in ("", "legacy", "none"):
#         freeze_teacher(model)
#         trainable, total, ratio = count_trainable_params(model)
#         print(f"[LatentSight][Trainable] legacy mode trainable={trainable:,}/{total:,} ({ratio:.4%})")
#         return trainable, total, ratio

#     for p in model.parameters():
#         p.requires_grad = False
#     freeze_teacher(model)

#     new_module_keywords = (
#         "student_world_adapter",
#         "structural_world_distill_loss",
#         "teacher_to_token_projector",
#         "world_condition_projector",
#         "world_condition_gate",
#         "world_denoise_modulator",
#     )
#     planner_keywords = ("action_head",)
#     vlm_keywords = ("backbone",)
#     lora_keywords = ("lora_A", "lora_B")

#     if mode == "projector_only":
#         _set_named_modules_trainable(model, new_module_keywords, True)
#     elif mode == "vlm_lora":
#         lora_cfg = getattr(model, "lora_cfg", {}) or {}
#         inject_lora(
#             getattr(getattr(model, "backbone", None), "model", getattr(model, "backbone", None)),
#             target_keywords=lora_cfg.get("target_modules"),
#             r=int(lora_cfg.get("r", 8)),
#             alpha=int(lora_cfg.get("alpha", 16)),
#             dropout=float(lora_cfg.get("dropout", 0.05)),
#         )
#         _set_named_modules_trainable(model, new_module_keywords + lora_keywords, True)
#     elif mode == "low_lr_full_finetune":
#         _set_named_modules_trainable(model, new_module_keywords + planner_keywords + vlm_keywords, True)
#     else:
#         raise ValueError(f"Unsupported LatentSight trainable mode: {mode}")

#     freeze_teacher(model)
#     _print_module_grad_state(model)
#     trainable, total, ratio = count_trainable_params(model)
#     print(f"[LatentSight][Trainable] mode={mode} total_trainable={trainable:,}/{total:,} ({ratio:.4%})")
#     return trainable, total, ratio


# def grad_norm_by_keywords(model: nn.Module, keywords: Sequence[str]) -> torch.Tensor:
#     device = next(model.parameters()).device
#     total = torch.zeros((), device=device)
#     for name, p in model.named_parameters():
#         if p.grad is not None and any(k in name for k in keywords):
#             total = total + p.grad.detach().float().norm(2).pow(2)
#     return total.sqrt()


# def build_latentsight_optimizer_groups(model: nn.Module, base_lr: float, weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
#     specs = [
#         ("projector", ("teacher_to_token_projector", "structural_world_distill_loss"), getattr(model, "projector_lr", 1e-4)),
#         ("future_queries", ("future_queries", "query", "student_world_adapter"), getattr(model, "future_query_lr", 1e-4)),
#         ("planner_condition", ("world_condition", "world_denoise_modulator"), getattr(model, "planner_condition_lr", 5e-5)),
#         ("lora", ("lora_A", "lora_B"), getattr(model, "lora_lr", 1e-4)),
#         ("vlm", ("backbone",), getattr(model, "vlm_lr", 5e-6)),
#         ("planner_head", ("action_head",), getattr(model, "planner_head_lr", 5e-5)),
#     ]
#     used = set()
#     groups = []
#     named = list(model.named_parameters())
#     for group_name, keywords, lr in specs:
#         params = [(n, p) for n, p in named if p.requires_grad and id(p) not in used and any(k in n for k in keywords)]
#         if not params:
#             continue
#         for _, p in params:
#             used.add(id(p))
#         groups.append({"params": [p for _, p in params], "lr": float(lr), "weight_decay": weight_decay, "name": group_name, "param_names": [n for n, _ in params]})
#     leftovers = [(n, p) for n, p in named if p.requires_grad and id(p) not in used]
#     if leftovers:
#         groups.append({"params": [p for _, p in leftovers], "lr": base_lr, "weight_decay": weight_decay, "name": "other", "param_names": [n for n, _ in leftovers]})
#     for g in groups:
#         nparams = sum(p.numel() for p in g["params"])
#         print(f"[LatentSight][Optimizer] group={g['name']} lr={g['lr']:.3g} params={nparams:,} sample={g['param_names'][:8]}")
#     return groups


# def sanity_check_latentsight(agent: nn.Module) -> Dict[str, Any]:
#     logs: Dict[str, Any] = {}
#     for key in ("last_teacher_latent", "last_z_teacher", "last_h_future"):
#         value = getattr(agent, key, None)
#         if isinstance(value, torch.Tensor):
#             logs[key] = {"shape": tuple(value.shape), "mean": float(value.detach().float().mean()), "std": float(value.detach().float().std()), "norm": float(value.detach().float().norm(dim=-1).mean())}
#     teacher = getattr(agent, "sgdrive_teacher", None)
#     if teacher is not None:
#         logs["teacher_has_grad"] = any(p.grad is not None for p in teacher.parameters())
#     logs["projector_grad_norm"] = float(grad_norm_by_keywords(agent, ["teacher_to_token_projector", "structural_world_distill_loss"]))
#     logs["lora_grad_norm"] = float(grad_norm_by_keywords(agent, ["lora_A", "lora_B"]))
#     print(f"[LatentSight][Sanity] {logs}")
#     return logs


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for an existing nn.Linear layer.

    The wrapped base layer remains frozen; only lora_A/lora_B are trainable.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # The wrapped VLM layer is often already on CUDA and may use bf16.
        # Newly-created LoRA modules default to CPU/fp32, so place them beside
        # the base layer immediately. Keep LoRA weights in fp32 for stable
        # optimization; forward casts the activation only for the LoRA branch.
        base_weight = self.base.weight
        self.lora_A.to(device=base_weight.device, dtype=torch.float32)
        self.lora_B.to(device=base_weight.device, dtype=torch.float32)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x).to(device=self.lora_A.weight.device, dtype=self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_in)) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype, device=base_out.device)


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    module: Optional[nn.Module],
    target_keywords: Optional[Sequence[str]] = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> List[str]:
    if module is None:
        return []
    target_keywords = tuple(target_keywords or LORA_TARGET_SUFFIXES)
    injected: List[str] = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
            continue
        if not any(name.endswith(k) or k in name for k in target_keywords):
            continue
        parent, attr = _get_parent_module(module, name)
        setattr(parent, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        injected.append(name)
    print(f"[LatentSight][LoRA] injected {len(injected)} Linear layers; sample={injected[:20]}")
    return injected


def freeze_teacher(model: nn.Module) -> None:
    teacher = getattr(model, "sgdrive_teacher", None)
    if teacher is None:
        return
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_agent = getattr(teacher, "teacher_agent", None)
    if teacher_agent is not None:
        teacher_agent.eval()
        for p in teacher_agent.parameters():
            p.requires_grad = False


def _set_named_modules_trainable(model: nn.Module, keywords: Sequence[str], trainable: bool = True) -> None:
    for name, p in model.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad = trainable

def _get_nested_module(root: nn.Module, module_path: str) -> Optional[nn.Module]:
    current: Any = root
    for part in module_path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, nn.Module) else None


def _set_named_submodules_trainable(model: nn.Module, module_paths: Sequence[str], trainable: bool = True) -> List[str]:
    matched: List[str] = []
    for module_path in module_paths:
        module = _get_nested_module(model, module_path)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = trainable
        matched.append(module_path)
    return matched


def _print_module_grad_state(model: nn.Module) -> None:
    rows = []
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if total:
            rows.append((name, trainable, total, trainable / total))
    for name, trainable, total, ratio in rows:
        print(f"[LatentSight][Trainable] module={name:<32} trainable={trainable:,}/{total:,} ({ratio:.4%})")


def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total, trainable / max(total, 1)


def set_v2_finetune_mode(model: nn.Module, mode: str, cfg: Optional[Any] = None) -> Tuple[int, int, float]:
    """Central trainable-module switch for generic LatentSight update strategies.

    Keep this function about parameter-update policy only (for example LoRA vs.
    low-LR full finetune). Experiment structure such as B1/B2/B3/B4 is controlled
    by YAML model/forward/loss flags, not by adding experiment-specific modes.
    """

    mode = (mode or "legacy").replace("-", "_").lower()
    if mode in ("", "legacy", "none"):
        freeze_teacher(model)
        trainable, total, ratio = count_trainable_params(model)
        print(f"[LatentSight][Trainable] legacy mode trainable={trainable:,}/{total:,} ({ratio:.4%})")
        return trainable, total, ratio

    for p in model.parameters():
        p.requires_grad = False
    freeze_teacher(model)

    future_query_keywords = ("student_world_adapter",)
    distill_projector_keywords = ("structural_world_distill_loss", "teacher_to_token_projector")
    planner_condition_keywords = ("world_condition_projector", "world_condition_gate", "world_denoise_modulator")
    new_module_keywords = future_query_keywords + distill_projector_keywords + planner_condition_keywords
    planner_keywords = ("action_head",)
    vlm_keywords = ("backbone",)

    llm_module_paths = ("backbone.model.language_model",)
    planner_module_paths = ("action_head",)
    vit_module_paths = ("backbone.model.vision_model",)
    vlm_projector_module_paths = ("backbone.model.mlp1",)

    lora_keywords = ("lora_A", "lora_B")

    if mode == "projector_only":
        # Distillation/projector pretraining only. Teacher stays frozen.
        _set_named_modules_trainable(model, distill_projector_keywords, True)
    elif mode == "adapter_only":
        # Lightweight student-side adapters/carriers only; no VLM/planner update.
        _set_named_modules_trainable(model, future_query_keywords + planner_condition_keywords, True)
    elif mode == "vlm_lora":
        lora_cfg = getattr(model, "lora_cfg", {}) or {}
        inject_lora(
            getattr(getattr(model, "backbone", None), "model", getattr(model, "backbone", None)),
            target_keywords=lora_cfg.get("target_modules"),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
        )
        _set_named_modules_trainable(model, new_module_keywords + lora_keywords, True)
    elif mode == "low_lr_full_finetune":
        # Generic full-student policy: train VLM/planner plus whichever optional
        # query/projector/modulation modules were actually constructed by YAML.
        _set_named_modules_trainable(model, new_module_keywords + planner_keywords + vlm_keywords, True)
    
    # elif mode in ("llm_planner_finetune", "freeze_vit_projector_train_llm_planner"):
    #     # Exact module paths for the current InternVL-based ReCogDrive backbone:
    #     # ReCogDriveAgent.backbone -> RecogDriveBackbone.model -> InternVLChatModel.
    #     # We fail fast if the required LLM/planner modules are absent instead of
    #     # silently training only a subset due to a stale keyword.
    #     matched_llm = _set_named_submodules_trainable(model, llm_module_paths, True)
    #     matched_planner = _set_named_submodules_trainable(model, planner_module_paths, True)
    #     _set_named_submodules_trainable(model, vit_module_paths + vlm_projector_module_paths, False)
    #     _set_named_modules_trainable(model, new_module_keywords, False)
    #     if not matched_llm:
    #         raise ValueError(
    #             "LatentSight mode freeze_vit_projector_train_llm_planner requires "
    #             "InternVL language module path 'backbone.model.language_model', "
    #             "but it was not found. Check RecogDriveBackbone/AutoModel module names."
    #         )
    #     if not matched_planner:
    #         raise ValueError("LatentSight mode freeze_vit_projector_train_llm_planner requires planner module 'action_head'.")
    #     print(f"[LatentSight][Trainable] llm/planner mode matched_llm={matched_llm} matched_planner={matched_planner}")

    elif mode in ("llm_planner_finetune", "freeze_vit_projector_train_llm_planner"):
        # Freeze only the visual side of the VLM (ViT + visual-to-LLM projector).
        # Keep LLM, planner, and any enabled LatentSight carrier/condition/distill
        # modules trainable so B2 query carriers can still learn from trajectory loss.
        _set_named_modules_trainable(model, new_module_keywords + planner_keywords + vlm_keywords, True)
        matched_llm = _set_named_submodules_trainable(model, llm_module_paths, True)
        matched_planner = _set_named_submodules_trainable(model, planner_module_paths, True)
        frozen_visual = _set_named_submodules_trainable(model, vit_module_paths + vlm_projector_module_paths, False)
        if not matched_llm:
            raise ValueError(
                "LatentSight mode freeze_vit_projector_train_llm_planner requires "
                "InternVL language module path 'backbone.model.language_model', "
                "but it was not found. Check RecogDriveBackbone/AutoModel module names."
            )
        if not matched_planner:
            raise ValueError("LatentSight mode freeze_vit_projector_train_llm_planner requires planner module 'action_head'.")
        print(
            f"[LatentSight][Trainable] freeze visual mode matched_llm={matched_llm} "
            f"matched_planner={matched_planner} frozen_visual={frozen_visual}"
        )
    
    else:
        raise ValueError(
            f"Unsupported LatentSight trainable mode: {mode}. "
            #"Use one of: projector_only, adapter_only, vlm_lora, low_lr_full_finetune."
            "Use one of: projector_only, adapter_only, vlm_lora, "
            "low_lr_full_finetune, llm_planner_finetune, "
            "freeze_vit_projector_train_llm_planner."
        )

    freeze_teacher(model)
    _print_module_grad_state(model)
    trainable, total, ratio = count_trainable_params(model)
    print(f"[LatentSight][Trainable] mode={mode} total_trainable={trainable:,}/{total:,} ({ratio:.4%})")
    return trainable, total, ratio


def grad_norm_by_keywords(model: nn.Module, keywords: Sequence[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if p.grad is not None and any(k in name for k in keywords):
            total = total + p.grad.detach().float().norm(2).pow(2)
    return total.sqrt()


def build_latentsight_optimizer_groups(model: nn.Module, base_lr: float, weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
    specs = [
        ("projector", ("teacher_to_token_projector", "structural_world_distill_loss"), getattr(model, "projector_lr", 1e-4)),
        ("future_queries", ("future_queries", "student_world_adapter"), getattr(model, "future_query_lr", 1e-4)),
        ("planner_condition", ("world_condition", "world_denoise_modulator"), getattr(model, "planner_condition_lr", 5e-5)),
        ("lora", ("lora_A", "lora_B"), getattr(model, "lora_lr", 1e-4)),
        ("llm", ("backbone.model.language_model",), getattr(model, "llm_lr", getattr(model, "vlm_lr", 5e-6))),
        ("vlm", ("backbone",), getattr(model, "vlm_lr", 5e-6)),
        ("planner_head", ("action_head",), getattr(model, "planner_head_lr", 5e-5)),
    ]
    used = set()
    groups = []
    named = list(model.named_parameters())
    for group_name, keywords, lr in specs:
        params = [(n, p) for n, p in named if p.requires_grad and id(p) not in used and any(k in n for k in keywords)]
        if not params:
            continue
        for _, p in params:
            used.add(id(p))
        groups.append({"params": [p for _, p in params], "lr": float(lr), "weight_decay": weight_decay, "name": group_name, "param_names": [n for n, _ in params]})
    leftovers = [(n, p) for n, p in named if p.requires_grad and id(p) not in used]
    if leftovers:
        groups.append({"params": [p for _, p in leftovers], "lr": base_lr, "weight_decay": weight_decay, "name": "other", "param_names": [n for n, _ in leftovers]})
    for g in groups:
        nparams = sum(p.numel() for p in g["params"])
        print(f"[LatentSight][Optimizer] group={g['name']} lr={g['lr']:.3g} params={nparams:,} sample={g['param_names'][:8]}")
    return groups


def sanity_check_latentsight(agent: nn.Module) -> Dict[str, Any]:
    logs: Dict[str, Any] = {}
    for key in ("last_teacher_latent", "last_z_teacher", "last_h_future"):
        value = getattr(agent, key, None)
        if isinstance(value, torch.Tensor):
            logs[key] = {"shape": tuple(value.shape), "mean": float(value.detach().float().mean()), "std": float(value.detach().float().std()), "norm": float(value.detach().float().norm(dim=-1).mean())}
    teacher = getattr(agent, "sgdrive_teacher", None)
    if teacher is not None:
        logs["teacher_has_grad"] = any(p.grad is not None for p in teacher.parameters())
    logs["projector_grad_norm"] = float(grad_norm_by_keywords(agent, ["teacher_to_token_projector", "structural_world_distill_loss"]))
    logs["lora_grad_norm"] = float(grad_norm_by_keywords(agent, ["lora_A", "lora_B"]))
    print(f"[LatentSight][Sanity] {logs}")
    return logs


# Backward-compatible public name used by older LatentSight configs.
def set_latentsight_trainable(model: nn.Module, mode: str, cfg: Optional[Any] = None) -> Tuple[int, int, float]:
    return set_v2_finetune_mode(model, mode, cfg)