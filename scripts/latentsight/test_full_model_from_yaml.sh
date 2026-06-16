#!/usr/bin/env bash
set -euo pipefail

# Evaluate a full ReCogDrive/LatentSight checkpoint with the agent topology
# defined by a training YAML (B1/B2/B3/B4 or any custom YAML). This is useful
# when full finetuning saves a single Lightning checkpoint and the eval-time
# agent must be reconstructed with the same YAML switches before loading weights.

REPO_ROOT=${REPO_ROOT:-/workspace/LD}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/recogdrive}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-${REPO_ROOT}/work_dirs}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

# Select one of B1/B2/B3/B3_DREAM/B3_FULL/B4/B5, or pass YAML_CONFIG directly for a custom YAML.

EXPERIMENT=${EXPERIMENT:-B1}
if [[ -z "${YAML_CONFIG:-}" ]]; then
  case "${EXPERIMENT}" in
    B1) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/v2/b1_student_low_lr_full_finetune.yaml ;;
    B2) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/v2/b2_query_carrier_low_lr_full_finetune.yaml ;;
    B3) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/v2/b3_teacher_future_distill_single_condition.yaml ;;
    #B4) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/v2/b4_teacher_distill_deep_modulation.yaml ;;
    B3_DREAM) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/b3_freeze_vit_dream_distill.yaml ;;
    B3_FULL) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/b3_freeze_vit_full_distill.yaml ;;
    B4) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/b4_freeze_vit_deep_modulation.yaml ;;
    B5) YAML_CONFIG=${REPO_ROOT}/configs/latentsight/b5_freeze_vit_three_stage_distill.yaml ;;
    *)
      #echo "[LatentSight][Eval][ERROR] Unsupported EXPERIMENT=${EXPERIMENT}. Use B1/B2/B3/B4 or set YAML_CONFIG=/path/to/config.yaml." >&2
      echo "[LatentSight][Eval][ERROR] Unsupported EXPERIMENT=${EXPERIMENT}. Use B1/B2/B3/B3_DREAM/B3_FULL/B4/B5 or set YAML_CONFIG=/path/to/config.yaml." >&2
      exit 2
      ;;
  esac
fi

# YAML_CONFIG can be any training YAML with an `agent:` block. The script parses
# the agent block and forwards the architecture/runtime switches that affect
# model construction and checkpoint loading to the PDM scoring config.
ckpt_var="CHECKPOINT_${EXPERIMENT}"
CHECKPOINT=${CHECKPOINT:-${!ckpt_var-}}
CHECKPOINT=${CHECKPOINT:-${STUDENT_CHECKPOINT:-}}
STUDENT_VLM_PATH=${STUDENT_VLM_PATH:-/path/to/ReCogDrive-VLM-2B}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}
CACHE_HIDDEN_STATE=${CACHE_HIDDEN_STATE:-false}
CACHE_MODE=${CACHE_MODE:-false}
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/work_dirs/latentsight_v2/full_model_eval/${EXPERIMENT,,}_$(basename "${YAML_CONFIG}" .yaml)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-latentsight_v2_full_model_eval_${EXPERIMENT,,}_$(basename "${YAML_CONFIG}" .yaml)}
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29513}

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[LatentSight][Eval][ERROR] Please set CHECKPOINT=/path/to/full_model.ckpt (or STUDENT_CHECKPOINT)." >&2
  exit 2
fi
if [[ ! -f "${YAML_CONFIG}" ]]; then
  echo "[LatentSight][Eval][ERROR] YAML_CONFIG does not exist: ${YAML_CONFIG}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

mapfile -d '' YAML_AGENT_OVERRIDES < <(python - "${YAML_CONFIG}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/common/agent/recogdrive_agent.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
default_agent_path = Path(sys.argv[2])
data = yaml.safe_load(config_path.read_text())
agent = data.get("agent", {}) or {}
default_agent = yaml.safe_load(default_agent_path.read_text()) if default_agent_path.exists() else {}
default_keys = set(default_agent or {})

# Keep this list focused on eval-time model construction and forward behavior.
# Optimizer/trainer-only keys are intentionally not forwarded to PDM scoring.
keys = [
    "cam_type",
    "vlm_type",
    "dit_type",
    "sampling_method",
    "cache_mode",
    "cache_hidden_state",
    "vlm_size",
    "train_backbone",
    "grpo",
    "use_sgdrive_teacher",
    "sgdrive_teacher_config",
    "sgdrive_teacher_checkpoint",
    "sgdrive_teacher_vlm_path",
    "sgdrive_teacher_vlm_type",
    "sgdrive_teacher_feature_keys",
    "student_world_keys",
    "struct_distill_keys",
    "freeze_sgdrive_teacher",
    "teacher_eval_mode",
    "debug_print_teacher_shapes",
    "teacher_feature_source",
    "use_student_world_adapter",
    "student_world_dim",
    "student_world_num_tokens",
    "student_world_use_mlp",
    "student_world_use_cross_attn",
    "student_world_num_heads",
    "student_world_source",
    "student_world_dropout",
    "use_structural_world_distill",
    "struct_distill_loss_type",
    "struct_distill_weight",
    "scene_distill_weight",
    "agent_distill_weight",
    "goal_distill_weight",
    "dream_agent_distill_weight",
    "dream_scene_distill_weight",
    "detach_teacher",
    "normalize_distill_features",
    "use_world_tokens_as_planner_condition",
    "world_condition_fusion_type",
    "world_condition_detach",
    "world_condition_weight",
    "use_world_denoise_influence",
    "world_denoise_influence_type",
    "use_world_denoise_cross_attn",
    "use_world_denoise_film",
    "world_denoise_dim",
    "world_denoise_num_heads",
    "world_denoise_dropout",
    "world_denoise_use_scene",
    "world_denoise_use_agent",
    "world_denoise_use_goal",
    "world_denoise_pooling",
    "world_denoise_apply_layers",
    "world_denoise_apply_steps",
    "world_denoise_custom_steps",
    "world_denoise_residual_scale",
    "world_denoise_gate_init",
    "world_denoise_detach_world_tokens",
    "debug_world_denoise_influence",
    "projector_type",
    "hidden_loss_type",
    "teacher_latent_dim",
    "projector_num_heads",
    "projector_dropout",
    "experiment_tag",
]

def hydra_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return yaml.safe_dump(list(value), default_flow_style=True).strip()
    text = str(value)
    # Keep empty strings explicit so Hydra does not drop them.
    if text == "":
        return "''"
    return text

for key in keys:
    if key in agent:
        prefix = "" if key in default_keys else "++"
        sys.stdout.write(f"{prefix}agent.{key}={hydra_value(agent[key])}\0")
PY
)

BASE_OVERRIDES=(
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "output_dir=${WORK_DIR}"
  "experiment_name=${EXPERIMENT_NAME}"
  "agent=recogdrive_agent"
)

FORCE_RUNTIME_OVERRIDES=(
  "agent.checkpoint_path=${CHECKPOINT}"
  "agent.vlm_path=${STUDENT_VLM_PATH}"
  "agent.cache_hidden_state=${CACHE_HIDDEN_STATE}"
  "agent.cache_mode=${CACHE_MODE}"
  "agent.use_sgdrive_teacher=false"
  "agent.use_structural_world_distill=false"
  "agent.struct_distill_weight=0.0"
  "agent.scene_distill_weight=0.0"
  "agent.agent_distill_weight=0.0"
  "agent.goal_distill_weight=0.0"
)

echo "[LatentSight][Eval] experiment=${EXPERIMENT}"
echo "[LatentSight][Eval] yaml_config=${YAML_CONFIG}"
echo "[LatentSight][Eval] checkpoint=${CHECKPOINT}"
echo "[LatentSight][Eval] vlm_path=${STUDENT_VLM_PATH}"
echo "[LatentSight][Eval] split=${TRAIN_TEST_SPLIT} metric_cache_path=${METRIC_CACHE_PATH}"
echo "[LatentSight][Eval] cache_hidden_state=${CACHE_HIDDEN_STATE} cache_mode=${CACHE_MODE} (full model image/VLM path by default)"
echo "[LatentSight][Eval] work_dir=${WORK_DIR}"
echo "[LatentSight][Eval] teacher=false distillation=false"
echo "[LatentSight][Eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"
echo "[LatentSight][Eval] forwarding ${#YAML_AGENT_OVERRIDES[@]} agent construction overrides from YAML"

cd "${NAVSIM_DEVKIT_ROOT}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  navsim/planning/script/run_pdm_score_recogdrive.py \
  "${BASE_OVERRIDES[@]}" \
  "${YAML_AGENT_OVERRIDES[@]}" \
  "${FORCE_RUNTIME_OVERRIDES[@]}"
