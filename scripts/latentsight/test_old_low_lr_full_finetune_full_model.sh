#!/usr/bin/env bash
set -euo pipefail

# Temporary evaluator for checkpoints trained with:
#   recogdrive/navsim/planning/script/config/common/agent/low_lr_full_finetune.yaml
#
# This script intentionally does not reuse/modify test_full_model_from_yaml.sh.
# It reconstructs the student-side full LatentSight topology from that agent YAML
# (query carrier + planner condition + deep denoise modulation), disables
# teacher/distillation for inference, and then loads the full finetuned ckpt.

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

CHECKPOINT=${CHECKPOINT:-${STUDENT_CHECKPOINT:-}}
STUDENT_VLM_PATH=${STUDENT_VLM_PATH:-${VLM_PATH:-}}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/work_dirs/latentsight_v2/eval_old_low_lr_full_finetune}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-latentsight_v2_eval_old_low_lr_full_finetune}
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29517}

# These defaults mirror common/agent/low_lr_full_finetune.yaml. Override them
# only if the checkpoint was trained with different architecture values.
CAM_TYPE=${CAM_TYPE:-single}
VLM_TYPE=${VLM_TYPE:-internvl}
DIT_TYPE=${DIT_TYPE:-small}
VLM_SIZE=${VLM_SIZE:-small}
SAMPLING_METHOD=${SAMPLING_METHOD:-ddim}
STUDENT_WORLD_DIM=${STUDENT_WORLD_DIM:-256}
STUDENT_WORLD_NUM_TOKENS=${STUDENT_WORLD_NUM_TOKENS:-4}
STUDENT_WORLD_USE_MLP=${STUDENT_WORLD_USE_MLP:-true}
STUDENT_WORLD_USE_CROSS_ATTN=${STUDENT_WORLD_USE_CROSS_ATTN:-true}
STUDENT_WORLD_NUM_HEADS=${STUDENT_WORLD_NUM_HEADS:-4}
STUDENT_WORLD_SOURCE=${STUDENT_WORLD_SOURCE:-cognition_tokens}
STUDENT_WORLD_DROPOUT=${STUDENT_WORLD_DROPOUT:-0.05}
WORLD_CONDITION_FUSION_TYPE=${WORLD_CONDITION_FUSION_TYPE:-concat}
WORLD_CONDITION_DETACH=${WORLD_CONDITION_DETACH:-false}
WORLD_CONDITION_WEIGHT=${WORLD_CONDITION_WEIGHT:-1.0}
WORLD_DENOISE_INFLUENCE_TYPE=${WORLD_DENOISE_INFLUENCE_TYPE:-cross_attn_plus_film}
WORLD_DENOISE_DIM=${WORLD_DENOISE_DIM:-256}
WORLD_DENOISE_NUM_HEADS=${WORLD_DENOISE_NUM_HEADS:-4}
WORLD_DENOISE_DROPOUT=${WORLD_DENOISE_DROPOUT:-0.0}
WORLD_DENOISE_POOLING=${WORLD_DENOISE_POOLING:-mean}
WORLD_DENOISE_RESIDUAL_SCALE=${WORLD_DENOISE_RESIDUAL_SCALE:-1.0}
WORLD_DENOISE_GATE_INIT=${WORLD_DENOISE_GATE_INIT:--2.0}

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[LatentSight][EvalOld][ERROR] Please set CHECKPOINT=/path/to/full_model.ckpt." >&2
  exit 2
fi
if [[ -z "${STUDENT_VLM_PATH}" ]]; then
  echo "[LatentSight][EvalOld][ERROR] Please set STUDENT_VLM_PATH=/path/to/ReCogDrive-VLM-2B (or VLM_PATH)." >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

echo "[LatentSight][EvalOld] checkpoint=${CHECKPOINT}"
echo "[LatentSight][EvalOld] student_vlm_path=${STUDENT_VLM_PATH}"
echo "[LatentSight][EvalOld] split=${TRAIN_TEST_SPLIT} metric_cache_path=${METRIC_CACHE_PATH}"
echo "[LatentSight][EvalOld] work_dir=${WORK_DIR}"
echo "[LatentSight][EvalOld] full_model_eval=true cache_hidden_state=false cache_mode=false"
echo "[LatentSight][EvalOld] teacher=false distillation=false"
echo "[LatentSight][EvalOld] student_world_adapter=true tokens=${STUDENT_WORLD_NUM_TOKENS} cross_attn=${STUDENT_WORLD_USE_CROSS_ATTN}"
echo "[LatentSight][EvalOld] planner_condition=true deep_modulation=true type=${WORLD_DENOISE_INFLUENCE_TYPE}"
echo "[LatentSight][EvalOld] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"

cd "${NAVSIM_DEVKIT_ROOT}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  navsim/planning/script/run_pdm_score_recogdrive.py \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  output_dir="${WORK_DIR}" \
  experiment_name="${EXPERIMENT_NAME}" \
  agent=recogdrive_agent \
  agent.checkpoint_path="${CHECKPOINT}" \
  agent.vlm_path="${STUDENT_VLM_PATH}" \
  agent.cam_type="${CAM_TYPE}" \
  agent.vlm_type="${VLM_TYPE}" \
  agent.dit_type="${DIT_TYPE}" \
  agent.vlm_size="${VLM_SIZE}" \
  agent.sampling_method="${SAMPLING_METHOD}" \
  agent.cache_hidden_state=false \
  agent.cache_mode=false \
  agent.train_backbone=false \
  agent.grpo=false \
  agent.use_sgdrive_teacher=false \
  agent.use_structural_world_distill=false \
  agent.struct_distill_weight=0.0 \
  agent.scene_distill_weight=0.0 \
  agent.agent_distill_weight=0.0 \
  agent.goal_distill_weight=0.0 \
  agent.use_student_world_adapter=true \
  agent.student_world_dim="${STUDENT_WORLD_DIM}" \
  agent.student_world_num_tokens="${STUDENT_WORLD_NUM_TOKENS}" \
  agent.student_world_use_mlp="${STUDENT_WORLD_USE_MLP}" \
  agent.student_world_use_cross_attn="${STUDENT_WORLD_USE_CROSS_ATTN}" \
  ++agent.student_world_num_heads="${STUDENT_WORLD_NUM_HEADS}" \
  agent.student_world_source="${STUDENT_WORLD_SOURCE}" \
  agent.student_world_dropout="${STUDENT_WORLD_DROPOUT}" \
  agent.use_world_tokens_as_planner_condition=true \
  agent.world_condition_fusion_type="${WORLD_CONDITION_FUSION_TYPE}" \
  agent.world_condition_detach="${WORLD_CONDITION_DETACH}" \
  agent.world_condition_weight="${WORLD_CONDITION_WEIGHT}" \
  agent.use_world_denoise_influence=true \
  agent.world_denoise_influence_type="${WORLD_DENOISE_INFLUENCE_TYPE}" \
  agent.use_world_denoise_cross_attn=true \
  agent.use_world_denoise_film=true \
  agent.world_denoise_dim="${WORLD_DENOISE_DIM}" \
  agent.world_denoise_num_heads="${WORLD_DENOISE_NUM_HEADS}" \
  agent.world_denoise_dropout="${WORLD_DENOISE_DROPOUT}" \
  agent.world_denoise_use_scene=true \
  agent.world_denoise_use_agent=true \
  agent.world_denoise_use_goal=true \
  agent.world_denoise_pooling="${WORLD_DENOISE_POOLING}" \
  agent.world_denoise_apply_layers=[] \
  agent.world_denoise_apply_steps=all \
  agent.world_denoise_custom_steps=[] \
  agent.world_denoise_residual_scale="${WORLD_DENOISE_RESIDUAL_SCALE}" \
  agent.world_denoise_gate_init="${WORLD_DENOISE_GATE_INIT}" \
  agent.world_denoise_detach_world_tokens=false \
  agent.debug_world_denoise_influence=false
