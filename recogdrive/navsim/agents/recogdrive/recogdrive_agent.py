from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)

from .sgdrive_teacher import FrozenSGDriveTeacher
from .world_distill import StudentWorldAdapter, StructuralWorldDistillLoss
import ast
import numpy as np
from .utils.internvl_preprocess import load_image, build_transform, dynamic_preprocess

from .latentsight_training import (
    build_latentsight_optimizer_groups,
    set_latentsight_trainable,
)

from .plugins.three_stage_distill import build_imagination_distiller, ThreeStageImaginationDistiller

class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        use_sgdrive_teacher: bool = False,
        sgdrive_teacher_config: str = "",
        sgdrive_teacher_checkpoint: str = "",
        sgdrive_teacher_vlm_path: str = "",
        sgdrive_teacher_vlm_type: str = "",
        sgdrive_teacher_feature_keys: Optional[List[str]] = None,
        freeze_sgdrive_teacher: bool = True,
        teacher_eval_mode: bool = True,
        debug_print_teacher_shapes: bool = False,
        use_student_world_adapter: bool = False,
        student_world_dim: int = 256,
        student_world_num_tokens: int = 1,
        student_world_use_mlp: bool = True,
        student_world_use_cross_attn: bool = False,
        student_world_source: str = "cognition_tokens",
        student_world_dropout: float = 0.0,
        use_structural_world_distill: bool = False,
        struct_distill_loss_type: str = "cosine",
        struct_distill_weight: float = 1.0,
        scene_distill_weight: float = 1.0,
        agent_distill_weight: float = 1.0,
        goal_distill_weight: float = 1.0,
        detach_teacher: bool = True,
        normalize_distill_features: bool = True,
        use_world_tokens_as_planner_condition: bool = False,
        world_condition_fusion_type: str = "concat",
        world_condition_detach: bool = False,
        world_condition_weight: float = 1.0,
        teacher_feature_source: str = "cached",
        use_world_denoise_influence: bool = False,
        world_denoise_influence_type: str = "cross_attn",
        use_world_denoise_cross_attn: bool = False,
        use_world_denoise_film: bool = False,
        world_denoise_dim: int = 256,
        world_denoise_num_heads: int = 4,
        world_denoise_dropout: float = 0.0,
        world_denoise_use_scene: bool = True,
        world_denoise_use_agent: bool = True,
        world_denoise_use_goal: bool = True,
        world_denoise_pooling: str = "none",
        world_denoise_apply_layers: Optional[List[int]] = None,
        world_denoise_apply_steps: str = "all",
        world_denoise_custom_steps: Optional[List[int]] = None,
        world_denoise_residual_scale: float = 1.0,
        world_denoise_gate_init: float = -2.0,
        world_denoise_detach_world_tokens: bool = False,
        debug_world_denoise_influence: bool = False,
        latentsight_train_mode: str = "legacy",
        projector_type: str = "mlp",
        hidden_loss_type: Optional[str] = None,
        teacher_latent_dim: int = 256,
        projector_num_heads: int = 4,
        projector_dropout: float = 0.0,
        student_world_num_heads: int = 4,
        use_paramwise_optimizer: bool = False,
        projector_lr: float = 1e-4,
        future_query_lr: float = 1e-4,
        planner_condition_lr: float = 5e-5,
        lora_lr: float = 1e-4,
        vlm_lr: float = 5e-6,
        planner_head_lr: float = 5e-5,
        optimizer_weight_decay: float = 1e-4,
        scheduler_epochs: int = 200,
        scheduler_warmup_epochs: int = 3,
        enable_teacher: Optional[bool] = None,
        use_teacher: Optional[bool] = None,
        use_distill: Optional[bool] = None,
        use_teacher_distill: Optional[bool] = None,
        lambda_hidden: Optional[float] = None,
        use_future_queries: Optional[bool] = None,
        use_h_future: Optional[bool] = None,
        use_planner_modulation_from_future: Optional[bool] = None,

        use_three_stage_distill: bool = False,
        distill_stage1_epochs: int = 2,
        distill_stage1_iters: int = -1,
        distill_stage2_end_epochs: int = 7,
        lambda_init: float = 1.0,
        lambda_corr: float = 1.0,
        lambda_cons: float = 0.1,
        lambda_teacher_min: float = 0.0,
        teacher_decay_type: str = "linear",
        beta_cos: float = 0.5,
        use_soft_intervention: bool = True,
        use_uncertainty: bool = False,
        use_self_consistency: bool = False,
        use_counterfactual_gain: bool = False,
        use_failure_type_attribution: bool = False,

        experiment_tag: str = "",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        
        if enable_teacher is not None:
            use_sgdrive_teacher = bool(enable_teacher) and bool(use_sgdrive_teacher)
        if use_teacher is not None:
            use_sgdrive_teacher = bool(use_teacher) and bool(use_sgdrive_teacher)
        if use_distill is not None:
            use_structural_world_distill = bool(use_distill) and bool(use_structural_world_distill)
        if use_teacher_distill is not None:
            use_structural_world_distill = bool(use_teacher_distill) and bool(use_structural_world_distill)
        if lambda_hidden is not None and float(lambda_hidden) <= 0.0:
            use_structural_world_distill = False
            struct_distill_weight = 0.0
            scene_distill_weight = 0.0
            agent_distill_weight = 0.0
            goal_distill_weight = 0.0
        if use_future_queries is not None:
            use_student_world_adapter = bool(use_future_queries) and bool(use_student_world_adapter)
        if use_h_future is not None:
            use_student_world_adapter = bool(use_h_future) and bool(use_student_world_adapter)
        if use_planner_modulation_from_future is not None:
            use_world_tokens_as_planner_condition = bool(use_planner_modulation_from_future) and bool(use_world_tokens_as_planner_condition)
            use_world_denoise_influence = bool(use_planner_modulation_from_future) and bool(use_world_denoise_influence)
        
        super().__init__()

        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.use_sgdrive_teacher = use_sgdrive_teacher
        self.debug_print_teacher_shapes = debug_print_teacher_shapes
        self.sgdrive_teacher: Optional[FrozenSGDriveTeacher] = None
        self.use_student_world_adapter = use_student_world_adapter
        self.student_world_source = student_world_source
        self.use_structural_world_distill = use_structural_world_distill
        self.latest_loss_logs: Dict[str, torch.Tensor] = {}
        self.student_world_adapter: Optional[StudentWorldAdapter] = None
        self.structural_world_distill_loss: Optional[StructuralWorldDistillLoss] = None
        self.use_world_tokens_as_planner_condition = use_world_tokens_as_planner_condition
        self.world_condition_fusion_type = world_condition_fusion_type
        self.world_condition_detach = world_condition_detach
        self.world_condition_weight = world_condition_weight
        self.world_condition_projector: Optional[torch.nn.Module] = None
        self.world_condition_gate: Optional[torch.nn.Parameter] = None
        self.teacher_feature_source = teacher_feature_source
        self.latentsight_train_mode = latentsight_train_mode.replace("-", "_") if latentsight_train_mode else "legacy"
        self.projector_type = projector_type
        self.hidden_loss_type = hidden_loss_type or struct_distill_loss_type
        self.teacher_latent_dim = teacher_latent_dim
        self.projector_lr = projector_lr
        self.future_query_lr = future_query_lr
        self.planner_condition_lr = planner_condition_lr
        self.lora_lr = lora_lr
        self.vlm_lr = vlm_lr
        self.planner_head_lr = planner_head_lr
        #self.use_paramwise_optimizer = use_paramwise_optimizer or self.latentsight_train_mode in ["projector_only", "vlm_lora", "low_lr_full_finetune"]
        
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_epochs = scheduler_epochs
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.experiment_tag = experiment_tag or self.latentsight_train_mode.upper()
        self.teacher_enabled = bool(self.use_sgdrive_teacher)
        self.distill_enabled = bool(self.use_structural_world_distill)
        self.future_queries_enabled = bool(self.use_student_world_adapter)
        self.use_paramwise_optimizer = use_paramwise_optimizer or self.latentsight_train_mode in ["projector_only", "adapter_only", "vlm_lora", "low_lr_full_finetune"]
        
        self.lora_cfg = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_target_modules,
        }
        self.trainable_param_ratio = 0.0
        self.last_teacher_latent = None
        self.last_z_teacher = None
        self.last_h_future = None

        self.use_three_stage_distill = bool(use_three_stage_distill)
        self.current_distill_epoch = 0
        self.current_distill_iter = 0
        self.three_stage_distiller: Optional[ThreeStageImaginationDistiller] = None
        if self.use_three_stage_distill:
            self.three_stage_distiller = build_imagination_distiller(
                {
                    "use_three_stage_distill": True,
                    "stage1_epochs": distill_stage1_epochs,
                    "stage1_iters": distill_stage1_iters,
                    "stage2_end_epochs": distill_stage2_end_epochs,
                    "lambda_init": lambda_init,
                    "lambda_corr": lambda_corr,
                    "lambda_cons": lambda_cons,
                    "lambda_teacher_min": lambda_teacher_min,
                    "teacher_decay_type": teacher_decay_type,
                    "beta_cos": beta_cos,
                    "use_soft_intervention": use_soft_intervention,
                    "use_uncertainty": use_uncertainty,
                    "use_self_consistency": use_self_consistency,
                    "use_counterfactual_gain": use_counterfactual_gain,
                    "use_failure_type_attribution": use_failure_type_attribution,
                }
            )

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size
        vlm_feature_dim = 3584 if self.vlm_size == "large" else 1536

        cfg.use_world_denoise_influence = use_world_denoise_influence
        cfg.world_denoise_influence_type = world_denoise_influence_type
        cfg.use_world_denoise_cross_attn = use_world_denoise_cross_attn
        cfg.use_world_denoise_film = use_world_denoise_film
        cfg.world_denoise_dim = world_denoise_dim
        cfg.world_denoise_num_heads = world_denoise_num_heads
        cfg.world_denoise_dropout = world_denoise_dropout
        cfg.world_denoise_use_scene = world_denoise_use_scene
        cfg.world_denoise_use_agent = world_denoise_use_agent
        cfg.world_denoise_use_goal = world_denoise_use_goal
        cfg.world_denoise_pooling = world_denoise_pooling
        cfg.world_denoise_apply_layers = world_denoise_apply_layers or []
        cfg.world_denoise_apply_steps = world_denoise_apply_steps
        cfg.world_denoise_custom_steps = world_denoise_custom_steps or []
        cfg.world_denoise_residual_scale = world_denoise_residual_scale
        cfg.world_denoise_gate_init = world_denoise_gate_init
        cfg.world_denoise_detach_world_tokens = world_denoise_detach_world_tokens
        cfg.debug_world_denoise_influence = debug_world_denoise_influence

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        
        if self.use_sgdrive_teacher:
            self.sgdrive_teacher = FrozenSGDriveTeacher(
                sgdrive_teacher_config=sgdrive_teacher_config,
                sgdrive_teacher_checkpoint=sgdrive_teacher_checkpoint,
                sgdrive_teacher_vlm_path=sgdrive_teacher_vlm_path or vlm_path or "",
                sgdrive_teacher_vlm_type=sgdrive_teacher_vlm_type or vlm_type or "",
                sgdrive_teacher_feature_keys=sgdrive_teacher_feature_keys or ["scene", "agent", "goal"],
                freeze_sgdrive_teacher=freeze_sgdrive_teacher,
                teacher_eval_mode=teacher_eval_mode,
                debug_print_teacher_shapes=debug_print_teacher_shapes,
            )
        if self.use_student_world_adapter:
            self.student_world_adapter = StudentWorldAdapter(
                input_dim=vlm_feature_dim,
                student_world_dim=student_world_dim,
                student_world_num_tokens=student_world_num_tokens,
                student_world_use_mlp=student_world_use_mlp,
                student_world_dropout=student_world_dropout,
                student_world_use_cross_attn=student_world_use_cross_attn,
                student_world_num_heads=student_world_num_heads,
            ).cuda()
        if self.use_world_tokens_as_planner_condition:
            self.world_condition_projector = torch.nn.Linear(student_world_dim, vlm_feature_dim).cuda()
            if world_condition_fusion_type == "gated_add":
                self.world_condition_gate = torch.nn.Parameter(torch.tensor(1.0, device=self.action_head.feature_encoder.weight.device))
            elif world_condition_fusion_type not in ["concat", "gated_add", "cross_attn"]:
                raise ValueError(f"Unsupported world_condition_fusion_type: {world_condition_fusion_type}")
        if self.use_structural_world_distill:
            self.structural_world_distill_loss = StructuralWorldDistillLoss(
                student_world_dim=student_world_dim,
                #struct_distill_loss_type=struct_distill_loss_type,
                struct_distill_loss_type=self.hidden_loss_type,
                struct_distill_weight=struct_distill_weight,
                scene_distill_weight=scene_distill_weight,
                agent_distill_weight=agent_distill_weight,
                goal_distill_weight=goal_distill_weight,
                detach_teacher=detach_teacher,
                normalize_distill_features=normalize_distill_features,
                student_world_num_tokens=student_world_num_tokens,
                projector_type=projector_type,
                teacher_latent_dim=teacher_latent_dim,
                projector_num_heads=projector_num_heads,
                projector_dropout=projector_dropout,
            ).cuda()

        trainable, total, ratio = set_latentsight_trainable(self, self.latentsight_train_mode, None)
        self.trainable_param_ratio = ratio

        print(
            f"[LatentSight][Experiment] name={self.experiment_tag} "
            f"teacher={self.teacher_enabled} future_queries={self.future_queries_enabled} "
            f"distillation={self.distill_enabled} trainable={trainable:,}/{total:,} ({ratio:.4%})"
        )

        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__
    
    @staticmethod
    def _load_trusted_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Load a project checkpoint saved by Lightning/torch.

        PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
        Our Lightning checkpoints may contain OmegaConf objects (for example
        ``ListConfig``) in metadata, so load trusted local training checkpoints
        with ``weights_only=False`` and fall back for older PyTorch versions.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_path!r} does not contain a 'state_dict' entry.")
        return checkpoint["state_dict"]

    def initialize(self) -> None:
        if self.checkpoint_path:
            #ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            ckpt = self._load_trusted_checkpoint(self.checkpoint_path)
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            
            # image_path_tensor = features["image_path_tensor"]
            # if image_path_tensor.ndim == 1:
            #     image_path_tensor = image_path_tensor.unsqueeze(0)
            # image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            # pixel_values_list = [load_image(path) for path in image_paths]

            if "pixel_values" in features:
                pixel_values = features["pixel_values"]
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                pixel_values_list = [pv for pv in pixel_values]
            else:
                image_path_tensor = features["image_path_tensor"]
                if image_path_tensor.ndim == 1:
                    image_path_tensor = image_path_tensor.unsqueeze(0)
                image_paths = self._decode_paths_from_tensor(image_path_tensor)
                pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
            

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)

        planner_world_tokens = None
        if self.use_student_world_adapter and self.student_world_adapter is not None:
            student_world = self.student_world_adapter(last_hidden_state)
            planner_world_tokens = torch.cat([student_world["scene"], student_world["agent"], student_world["goal"]], dim=1)
            if self.world_condition_detach:
                planner_world_tokens = planner_world_tokens.detach()
            if self.debug_print_teacher_shapes and self.training:
                print(f"[ReCogDriveAgent] student world token shapes scene={tuple(student_world['scene'].shape)} agent={tuple(student_world['agent'].shape)} goal={tuple(student_world['goal'].shape)}")

            if self.use_world_tokens_as_planner_condition and self.world_condition_projector is not None:
                world_cond = self.world_condition_projector(planner_world_tokens.to(model_dtype))
                if self.world_condition_fusion_type == "concat" or self.world_condition_fusion_type == "cross_attn":
                    # cross_attn is currently aliased to concat as a stable fallback.
                    last_hidden_state = torch.cat([last_hidden_state, self.world_condition_weight * world_cond], dim=1)
                elif self.world_condition_fusion_type == "gated_add":
                    world_summary = world_cond.mean(dim=1, keepdim=True).expand(-1, last_hidden_state.shape[1], -1)
                    gate = torch.sigmoid(self.world_condition_gate).to(world_summary.dtype) if self.world_condition_gate is not None else 1.0
                    last_hidden_state = last_hidden_state + self.world_condition_weight * gate * world_summary
                if self.debug_print_teacher_shapes and self.training:
                    print(f"[ReCogDriveAgent] planner conditioned with world tokens: fusion={self.world_condition_fusion_type}, cog={tuple(last_hidden_state.shape)}, world={tuple(world_cond.shape)}")

        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training and self.use_sgdrive_teacher and self.sgdrive_teacher is not None:
            #teacher_outputs = self.sgdrive_teacher(features, targets=targets, tokens_list=tokens_list, teacher_feature_source=self.teacher_feature_source)
            
            self.sgdrive_teacher.eval()
            for _p in self.sgdrive_teacher.parameters():
                _p.requires_grad = False
            with torch.no_grad():
                teacher_outputs = self.sgdrive_teacher(features, targets=targets, tokens_list=tokens_list, teacher_feature_source=self.teacher_feature_source)
            
            # Training-only debug outputs; loss integration happens in later patches.
            features["teacher_scene"] = teacher_outputs.get("scene")
            features["teacher_agent"] = teacher_outputs.get("agent")
            features["teacher_goal"] = teacher_outputs.get("goal")
            features["teacher_world"] = teacher_outputs.get("world")

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            #return self.action_head(last_hidden_state, action_inputs)
            outputs = self.action_head(last_hidden_state, action_inputs, world_tokens=student_world if self.use_student_world_adapter and self.student_world_adapter is not None else None)
            # self.latest_loss_logs = {}
            # self.latest_loss_logs["loss_plan"] = outputs.loss.detach()
            # self.latest_loss_logs["trainable_param_ratio"] = torch.tensor(self.trainable_param_ratio, device=outputs.loss.device)
            # self.latest_loss_logs["world_tokens_used_by_planner"] = torch.tensor(1.0 if self.use_world_tokens_as_planner_condition else 0.0, device=outputs.loss.device)
            
            # struct_logs["loss_plan"] = self.latest_loss_logs["loss_plan"]
            # struct_logs["trainable_param_ratio"] = self.latest_loss_logs["trainable_param_ratio"]
            # struct_logs["projector_grad_norm"] = torch.tensor(0.0, device=loss_struct_total.device)
            # # Save tensors for optional debug/sanity inspection.
            # self.last_h_future = next((v for v in student_world_for_loss.values() if isinstance(v, torch.Tensor)), None)
            # teacher_first = next((v for v in teacher_outputs.values() if isinstance(v, torch.Tensor)), None)
            # self.last_teacher_latent = teacher_first
            # if teacher_first is not None and self.structural_world_distill_loss is not None and self.last_h_future is not None:
            #     key_first = next((k for k, v in teacher_outputs.items() if isinstance(v, torch.Tensor)), "scene")
            #     self.last_z_teacher = self.structural_world_distill_loss.teacher_to_token_projector[key_first](teacher_first.to(dtype=self.last_h_future.dtype, device=self.last_h_future.device))
        
            # if self.use_structural_world_distill and self.training and self.use_sgdrive_teacher and self.sgdrive_teacher is not None and self.student_world_adapter is not None and self.structural_world_distill_loss is not None:
            #     teacher_outputs = {            
            
            zero = torch.tensor(0.0, device=outputs.loss.device)
            self.latest_loss_logs = {
                "loss_plan": outputs.loss.detach(),
                "total_loss": outputs.loss.detach(),
                "loss_hidden": zero,
                "trainable_param_ratio": torch.tensor(self.trainable_param_ratio, device=outputs.loss.device),
                "world_tokens_used_by_planner": torch.tensor(1.0 if self.use_world_tokens_as_planner_condition else 0.0, device=outputs.loss.device),
                "teacher_enabled": torch.tensor(1.0 if self.use_sgdrive_teacher else 0.0, device=outputs.loss.device),
                "distill_enabled": torch.tensor(1.0 if self.use_structural_world_distill else 0.0, device=outputs.loss.device),
            }
            if self.use_student_world_adapter and self.student_world_adapter is not None and 'student_world' in locals():
                self.last_h_future = next((v for v in student_world.values() if isinstance(v, torch.Tensor)), None)
                if self.last_h_future is not None:
                    self.latest_loss_logs["H_future_norm"] = self.last_h_future.detach().float().norm(dim=-1).mean()
            else:
                self.last_h_future = None

            if self.use_structural_world_distill and self.use_sgdrive_teacher and self.sgdrive_teacher is not None and self.student_world_adapter is not None and self.structural_world_distill_loss is not None:
                teacher_outputs_for_loss = {
                    "scene": features.get("teacher_scene", None),
                    "agent": features.get("teacher_agent", None),
                    "goal": features.get("teacher_goal", None),
                    "world": features.get("teacher_world", None),
                }
                # student_world_for_loss = student_world if self.use_student_world_adapter and self.student_world_adapter is not None else self.student_world_adapter(last_hidden_state)
                # loss_struct_total, struct_logs = self.structural_world_distill_loss(student_world_for_loss, teacher_outputs)
                # struct_logs["world_tokens_used_by_planner"] = torch.tensor(1.0 if self.use_world_tokens_as_planner_condition else 0.0, device=loss_struct_total.device)
                student_world_for_loss = student_world if 'student_world' in locals() else self.student_world_adapter(last_hidden_state)
                # loss_struct_total, struct_logs = self.structural_world_distill_loss(student_world_for_loss, teacher_outputs_for_loss)
                # outputs.loss = outputs.loss + loss_struct_total
                # struct_logs["loss_plan"] = self.latest_loss_logs["loss_plan"]
                # struct_logs["total_loss"] = outputs.loss.detach()
                # struct_logs["trainable_param_ratio"] = self.latest_loss_logs["trainable_param_ratio"]
                # struct_logs["world_tokens_used_by_planner"] = self.latest_loss_logs["world_tokens_used_by_planner"]
                # struct_logs["teacher_enabled"] = self.latest_loss_logs["teacher_enabled"]
                # struct_logs["distill_enabled"] = self.latest_loss_logs["distill_enabled"]
                # self.latest_loss_logs = struct_logs
                
                teacher_first = next((v for v in teacher_outputs_for_loss.values() if isinstance(v, torch.Tensor)), None)
                self.last_teacher_latent = teacher_first
                if teacher_first is not None and self.last_h_future is not None:
                    key_first = next((k for k, v in teacher_outputs_for_loss.items() if isinstance(v, torch.Tensor)), "scene")
                    #self.last_z_teacher = self.structural_world_distill_loss.teacher_to_token_projector[key_first](teacher_first.to(dtype=self.last_h_future.dtype, device=self.last_h_future.device))
                    self.last_z_teacher = self.structural_world_distill_loss.teacher_to_token_projector[key_first](
                        teacher_first.to(dtype=self.last_h_future.dtype, device=self.last_h_future.device)
                    )

                if self.use_three_stage_distill and self.three_stage_distiller is not None:
                    traj_pred = None
                    with torch.no_grad():
                        try:
                            sampled = self.action_head.get_action(
                                last_hidden_state.detach().to(model_dtype),
                                BatchFeature({
                                    "state": input_state.detach().to(model_dtype),
                                    "his_traj": history_trajectory_reshaped.detach().to(model_dtype),
                                    "status_feature": status_feature.detach().to(model_dtype),
                                }),
                                deterministic=True,
                                world_tokens=student_world_for_loss if self.use_student_world_adapter else None,
                            )
                            traj_pred = sampled.get("pred_traj", None)
                        except Exception as exc:  # keep the plugin non-invasive if sampling is unavailable
                            if self.debug_print_teacher_shapes:
                                print(f"[LatentSight][ThreeStage][WARN] skipped traj sampling for failure score: {exc}")
                    teacher_requires_grad_count = sum(
                        1 for p in self.sgdrive_teacher.parameters() if p.requires_grad
                    ) if self.sgdrive_teacher is not None else 0
                    distill_losses, distill_logs = self.three_stage_distiller(
                        h_future=self.last_h_future,
                        z_teacher=self.last_z_teacher,
                        traj_pred=traj_pred,
                        traj_gt=targets.get("trajectory") if isinstance(targets, dict) else None,
                        h_future_aug=features.get("H_future_aug", None),
                        h_future_ref=features.get("H_future_ref", None),
                        extras={**features, **(targets if isinstance(targets, dict) else {})},
                        epoch=int(getattr(self, "current_distill_epoch", 0)),
                        iteration=int(getattr(self, "current_distill_iter", 0)),
                        teacher_enabled=self.use_sgdrive_teacher,
                        teacher_requires_grad_count=teacher_requires_grad_count,
                    )
                    loss_three_stage = (
                        distill_losses["loss_hidden"]
                        + distill_losses["loss_corrective"]
                        + distill_losses["loss_consistency"]
                    )
                    outputs.loss = outputs.loss + loss_three_stage
                    distill_logs["loss_plan"] = self.latest_loss_logs["loss_plan"]
                    distill_logs["loss_total"] = outputs.loss.detach()
                    distill_logs["total_loss"] = outputs.loss.detach()
                    distill_logs["loss_hidden"] = distill_losses["loss_hidden"].detach()
                    distill_logs["loss_corrective_weighted"] = distill_losses["loss_corrective"].detach()
                    distill_logs["loss_consistency_weighted"] = distill_losses["loss_consistency"].detach()
                    distill_logs["trainable_param_ratio"] = self.latest_loss_logs["trainable_param_ratio"]
                    distill_logs["world_tokens_used_by_planner"] = self.latest_loss_logs["world_tokens_used_by_planner"]
                    distill_logs["distill_enabled"] = self.latest_loss_logs["distill_enabled"]
                    self.latest_loss_logs = distill_logs
                else:
                    loss_struct_total, struct_logs = self.structural_world_distill_loss(student_world_for_loss, teacher_outputs_for_loss)
                    outputs.loss = outputs.loss + loss_struct_total
                    struct_logs["loss_plan"] = self.latest_loss_logs["loss_plan"]
                    struct_logs["total_loss"] = outputs.loss.detach()
                    struct_logs["loss_total"] = outputs.loss.detach()
                    struct_logs["trainable_param_ratio"] = self.latest_loss_logs["trainable_param_ratio"]
                    struct_logs["world_tokens_used_by_planner"] = self.latest_loss_logs["world_tokens_used_by_planner"]
                    struct_logs["teacher_enabled"] = self.latest_loss_logs["teacher_enabled"]
                    struct_logs["distill_enabled"] = self.latest_loss_logs["distill_enabled"]
                    self.latest_loss_logs = struct_logs
            else:
                self.last_teacher_latent = None
                self.last_z_teacher = None
            return outputs
        
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        else: 
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    # def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
    #     optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

    #     if self.use_paramwise_optimizer:
    #         params = build_latentsight_optimizer_groups(self, base_lr=self._lr, weight_decay=1e-4)
    #         optimizer = build_from_configs(optim, optimizer_cfg, params=params)
    #         scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
    #         return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
    #     params = list(self.action_head.parameters())
    #     if self.backbone is not None and self.train_backbone:
    #         params += list(self.backbone.parameters())
        
    #     if self.student_world_adapter is not None:
    #         params += list(self.student_world_adapter.parameters())
    #     if self.structural_world_distill_loss is not None:
    #         params += list(self.structural_world_distill_loss.parameters())
    #     if self.world_condition_projector is not None:
    #         params += list(self.world_condition_projector.parameters())
    #     if self.world_condition_gate is not None:
    #         params += [self.world_condition_gate]

    #     optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
    #     if self.grpo:
    #         scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
    #     else:
    #         scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        #optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=self.optimizer_weight_decay, betas=(0.9, 0.95)))

        if self.use_paramwise_optimizer:
            #params = build_latentsight_optimizer_groups(self, base_lr=self._lr, weight_decay=1e-4)
            params = build_latentsight_optimizer_groups(self, base_lr=self._lr, weight_decay=self.optimizer_weight_decay)
            optimizer = build_from_configs(optim, optimizer_cfg, params=params)
            #scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=self.scheduler_epochs, warmup_epochs=self.scheduler_warmup_epochs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())
        
        if self.student_world_adapter is not None:
            params += list(self.student_world_adapter.parameters())
        if self.structural_world_distill_loss is not None:
            params += list(self.structural_world_distill_loss.parameters())
        if self.world_condition_projector is not None:
            params += list(self.world_condition_projector.parameters())
        if self.world_condition_gate is not None:
            params += [self.world_condition_gate]

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            #scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=self.scheduler_epochs, warmup_epochs=self.scheduler_warmup_epochs)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
