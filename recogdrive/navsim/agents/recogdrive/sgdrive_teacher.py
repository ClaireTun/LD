from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class FrozenSGDriveTeacher(torch.nn.Module):
    """Training-only frozen SGDrive teacher wrapper.

    Returns a dictionary with optional structural features:
    {
        "scene": Tensor | None,
        "agent": Tensor | None,
        "goal": Tensor | None,
        "world": Tensor | None,
    }
    """

    _FEATURE_MAP = {
        "scene": ["occ_hidden_state", "scene_hidden_state", "scene"],
        "agent": ["agent_hidden_state", "agent"],
        "goal": ["gp_hidden_state", "goal_hidden_state", "goal"],
        "world": ["dream_occ_hidden_state", "dream_agent_hidden_state", "world"],
    }

    def __init__(
        self,
        sgdrive_teacher_config: str,
        sgdrive_teacher_checkpoint: str = "",
        sgdrive_teacher_feature_keys: Optional[List[str]] = None,
        freeze_sgdrive_teacher: bool = True,
        teacher_eval_mode: bool = True,
        debug_print_teacher_shapes: bool = False,
    ) -> None:
        super().__init__()
        if not sgdrive_teacher_config:
            raise ValueError("sgdrive_teacher_config must be set when using SGDrive teacher.")

        self.sgdrive_teacher_config = sgdrive_teacher_config
        self.sgdrive_teacher_checkpoint = sgdrive_teacher_checkpoint
        self.feature_keys = sgdrive_teacher_feature_keys or ["scene", "agent", "goal"]
        self.freeze_sgdrive_teacher = freeze_sgdrive_teacher
        self.teacher_eval_mode = teacher_eval_mode
        self.debug_print_teacher_shapes = debug_print_teacher_shapes
        self._warned_keys = set()
        self._printed_shapes = False

        self.teacher_agent = self._build_teacher_agent()
        if self.teacher_eval_mode:
            self.teacher_agent.eval()
        if self.freeze_sgdrive_teacher:
            for p in self.teacher_agent.parameters():
                p.requires_grad = False

    def _build_teacher_agent(self):
        cfg = OmegaConf.load(self.sgdrive_teacher_config)
        agent = instantiate(cfg)
        if self.sgdrive_teacher_checkpoint:
            agent.checkpoint_path = self.sgdrive_teacher_checkpoint
        agent.initialize()
        return agent.cuda()

    def _extract_from_features(self, features: Dict[str, torch.Tensor], key: str):
        candidates = self._FEATURE_MAP.get(key, [key])
        for name in candidates:
            value = features.get(name, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _warn_once(self, key: str, msg: str) -> None:
        token = f"{key}:{msg}"
        if token not in self._warned_keys:
            print(f"[FrozenSGDriveTeacher][WARN] {msg}")
            self._warned_keys.add(token)

    def _extract_from_prediction(self, prediction: Any, key: str):
        if prediction is None:
            return None
        candidates = self._FEATURE_MAP.get(key, [key])
        if isinstance(prediction, dict):
            for name in candidates:
                value = prediction.get(name, None)
                if isinstance(value, torch.Tensor):
                    return value
        for name in candidates:
            value = getattr(prediction, name, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @torch.no_grad()
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        tokens_list: Optional[Any] = None,
        teacher_feature_source: str = "cached",
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.teacher_eval_mode:
            self.teacher_agent.eval()

        outputs: Dict[str, Optional[torch.Tensor]] = {"scene": None, "agent": None, "goal": None, "world": None}

        teacher_prediction = None
        if teacher_feature_source == "online":
            # Online mode: run frozen teacher forward first, then extract best-effort structural features.
            try:
                teacher_prediction = self.teacher_agent.forward(features, targets=targets, tokens_list=tokens_list)
            except Exception as e:
                self._warn_once("online", f"teacher online forward failed ({e}); fallback to cached feature extraction.")
        elif teacher_feature_source != "cached":
            self._warn_once("feature_source", f"unknown teacher_feature_source={teacher_feature_source}, fallback to cached")

        for key in outputs.keys():
            if key not in self.feature_keys and key != "world":
                continue
            tensor = None
            if teacher_feature_source == "online":
                tensor = self._extract_from_prediction(teacher_prediction, key)
            if tensor is None:
                tensor = self._extract_from_features(features, key)
            if tensor is None:
                self._warn_once(key, f"teacher feature '{key}' unavailable; returning None.")
            outputs[key] = tensor

        if self.debug_print_teacher_shapes and not self._printed_shapes:
            shape_info = {k: (None if v is None else tuple(v.shape)) for k, v in outputs.items()}
            print(f"[FrozenSGDriveTeacher] config={self.sgdrive_teacher_config}")
            print(f"[FrozenSGDriveTeacher] checkpoint={self.sgdrive_teacher_checkpoint}")
            print(f"[FrozenSGDriveTeacher] feature_source={teacher_feature_source}")
            print(f"[FrozenSGDriveTeacher] extracted feature shapes={shape_info}")
            self._printed_shapes = True

        return outputs
