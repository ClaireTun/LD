from __future__ import annotations

from types import SimpleNamespace
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
        "scene": ["occ_hidden_state", "occ_out", "scene_hidden_state", "scene"],
        "agent": ["agent_hidden_state", "agent_out", "agent"],
        "goal": ["gp_hidden_state", "gp_out", "goal_hidden_state", "goal"],
        "world": ["dream_occ_hidden_state", "dream_occ_out", "dream_agent_hidden_state", "dream_agent_out", "world"],
    }

    def __init__(
        self,
        sgdrive_teacher_config: str,
        sgdrive_teacher_checkpoint: str = "",
        sgdrive_teacher_feature_keys: Optional[List[str]] = None,
        freeze_sgdrive_teacher: bool = True,
        teacher_eval_mode: bool = True,
        debug_print_teacher_shapes: bool = False,
        teacher_feature_source: str = "cached",
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
        self.teacher_feature_source = teacher_feature_source
        self._warned_keys = set()
        self._printed_shapes = False

        # Cached distillation can read teacher features directly from the batch and
        # should not pay the cost or import risk of constructing SGDrive. Online
        # mode lazily initializes the teacher the first time it is needed.
        self.teacher_agent = None
        if self.teacher_feature_source == "online":
            self._ensure_teacher_agent()

    def _build_teacher_agent(self):
        cfg = OmegaConf.load(self.sgdrive_teacher_config)
        if self.teacher_feature_source == "online":
            # SGDriveAgent only builds its VLM backbone in no-cache mode. Force
            # the teacher config into that mode for online distillation, while
            # leaving cached distillation free of any SGDrive model construction.
            cfg.cache_hidden_state = False
            cfg.cache_mode = False
        agent = instantiate(cfg)
        if self.sgdrive_teacher_checkpoint:
            agent.checkpoint_path = self.sgdrive_teacher_checkpoint
        agent.initialize()
        return agent.cuda()

    def _ensure_teacher_agent(self):
        if self.teacher_agent is None:
            self.teacher_agent = self._build_teacher_agent()
            if self.teacher_eval_mode:
                self.teacher_agent.eval()
            if self.freeze_sgdrive_teacher:
                for p in self.teacher_agent.parameters():
                    p.requires_grad = False
        return self.teacher_agent

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

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = int(code.item())
                if code_item == 0:
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

    @staticmethod
    def _split_pixel_values(pixel_values: torch.Tensor) -> List[torch.Tensor]:
        if pixel_values.ndim == 3:
            return [pixel_values]
        if pixel_values.ndim == 4:
            # Ambiguous shape: [patches, C, H, W] for one sample, or
            # [B, C, H, W] for raw batched images. In this pipeline collate gives
            # [B, patches, C, H, W], so 4D is treated as one preprocessed sample.
            return [pixel_values]
        if pixel_values.ndim == 5:
            return [sample for sample in pixel_values]
        raise ValueError(f"Unsupported pixel_values shape for SGDrive teacher: {tuple(pixel_values.shape)}")

    def _pixel_values_from_features(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if "pixel_values" in features and isinstance(features["pixel_values"], torch.Tensor):
            return self._split_pixel_values(features["pixel_values"])

        image_path_tensor = features.get("image_path_tensor", None)
        if not isinstance(image_path_tensor, torch.Tensor):
            raise KeyError("SGDrive online teacher needs either features['pixel_values'] or features['image_path_tensor'].")
        if image_path_tensor.ndim == 1:
            image_path_tensor = image_path_tensor.unsqueeze(0)
        image_paths = self._decode_paths_from_tensor(image_path_tensor)
        from navsim.agents.sgdrive.utils.internvl_preprocess import load_image

        return [load_image(path, max_num=4) for path in image_paths]

    @staticmethod
    def _command_tensor(command: torch.Tensor) -> torch.Tensor:
        if command.ndim == 0:
            command = command.unsqueeze(0)
        return command.detach().float().cpu()

    def _ego_statuses_from_features(
        self,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        status_feature: Optional[torch.Tensor],
    ) -> List[Any]:
        command = self._command_tensor(high_command_one_hot)
        status = status_feature.detach().float().cpu() if isinstance(status_feature, torch.Tensor) else None
        velocity = status[3:5].tolist() if status is not None and status.numel() >= 5 else [0.0, 0.0]
        acceleration = status[5:7].tolist() if status is not None and status.numel() >= 7 else [0.0, 0.0]
        statuses = []
        history = history_trajectory.detach().float().cpu()
        for pose in history:
            statuses.append(
                SimpleNamespace(
                    driving_command=command.tolist(),
                    ego_acceleration=acceleration,
                    ego_pose=pose.tolist(),
                    ego_velocity=velocity,
                )
            )
        return statuses

    @staticmethod
    def _format_number(value: float, decimal_places: int = 2) -> str:
        return f"{value:+.{decimal_places}f}" if abs(round(value, decimal_places)) > 1e-2 else "0.0"

    def _build_question(self, history_trajectory: torch.Tensor, high_command_one_hot: torch.Tensor) -> str:
        navigation_commands = ["turn left", "go straight", "turn right"]
        command_idx = int(torch.argmax(high_command_one_hot.detach().cpu()).item())
        command_str = navigation_commands[command_idx] if command_idx < len(navigation_commands) else "unknown"
        history = history_trajectory.detach().float().cpu()
        history_str = " ".join(
            [
                f"   - t-{3 - i}: ({self._format_number(history[i, 0].item())}, "
                f"{self._format_number(history[i, 1].item())}, "
                f"{self._format_number(history[i, 2].item())})"
                for i in range(history.shape[0])
            ]
        )
        prompt = (
            "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{command_str.upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        return f"{prompt}{output_requirements}"

    def _run_online_backbone(self, features: Dict[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        teacher_agent = self._ensure_teacher_agent()
        backbone = getattr(teacher_agent, "backbone", None)
        if backbone is None:
            return teacher_agent.forward(features)

        pixel_values_list = self._pixel_values_from_features(features)
        history_trajectory = features["history_trajectory"]
        high_command_one_hot = features["high_command_one_hot"]
        status_feature = features.get("status_feature", None)
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        if isinstance(status_feature, torch.Tensor) and status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)

        batched_outputs: Dict[str, List[torch.Tensor]] = {}
        for idx, pixel_values in enumerate(pixel_values_list):
            sample_history = history_trajectory[idx]
            sample_command = high_command_one_hot[idx]
            sample_status = status_feature[idx] if isinstance(status_feature, torch.Tensor) else None
            pixel_values = pixel_values.cuda()
            question = self._build_question(sample_history, sample_command)
            ego_statuses = self._ego_statuses_from_features(sample_history, sample_command, sample_status)
            output = backbone(
                pixel_values,
                [question],
                num_patches_list=[pixel_values.shape[0]],
                ego_status=ego_statuses,
            )
            for key in ["occ_out", "agent_out", "gp_out", "dream_occ_out", "dream_agent_out"]:
                value = output.get(key, None) if isinstance(output, dict) else None
                if isinstance(value, torch.Tensor):
                    batched_outputs.setdefault(key, []).append(value)

        return {key: torch.cat(values, dim=0) for key, values in batched_outputs.items() if values}

    @torch.no_grad()
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        tokens_list: Optional[Any] = None,
        teacher_feature_source: str = "cached",
    ) -> Dict[str, Optional[torch.Tensor]]:
        outputs: Dict[str, Optional[torch.Tensor]] = {"scene": None, "agent": None, "goal": None, "world": None}

        self.teacher_feature_source = teacher_feature_source
        teacher_prediction = None
        if teacher_feature_source == "online":
            # Online mode: run the copied SGDrive teacher/backbone and then extract
            # structural world features. If online inference fails, keep training
            # alive by falling back to cached feature extraction.
            try:
                if self.teacher_eval_mode:
                    self._ensure_teacher_agent().eval()
                teacher_prediction = self._run_online_backbone(features)
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
