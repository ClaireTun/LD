#!/usr/bin/env bash

#sleep 52h

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ma-user/ipfs/tuntun/LSD}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/recogdrive}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-${REPO_ROOT}/exp}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/home/ma-user/ipfs/tuntun/data/navsim/dataset/maps}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/home/ma-user/ipfs/tuntun/data/navsim/dataset}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

CONFIG=${CONFIG:-${REPO_ROOT}/configs/latentsight/b3_teacher_future_distill_single_condition.yaml}
STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-}
STUDENT_VLM_PATH=${STUDENT_VLM_PATH:-/home/ma-user/ipfs/tuntun/SGDrive-main/ReCogDrive-VLM-2B}

SGDRIVE_TEACHER_CONFIG=${SGDRIVE_TEACHER_CONFIG:-/home/ma-user/ipfs/tuntun/LSD/SGDrive/navsim/planning/script/config/common/agent/sgdrive_agent.yaml}
SGDRIVE_TEACHER_CHECKPOINT=${SGDRIVE_TEACHER_CHECKPOINT:-/home/ma-user/ipfs/tuntun/SGDrive-main/premodel/stage2_h20_171.ckpt}
SGDRIVE_TEACHER_VLM_PATH=${SGDRIVE_TEACHER_VLM_PATH:-/home/ma-user/ipfs/tuntun/SGDrive-main/premodel/stage1}
SGDRIVE_TEACHER_VLM_TYPE=${SGDRIVE_TEACHER_VLM_TYPE:-internvl_wm}
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/exp/latentsight_v2/b3_teacher_future_distill_single_condition}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
GPUS=${GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29513}

export STUDENT_CHECKPOINT STUDENT_VLM_PATH SGDRIVE_TEACHER_CONFIG SGDRIVE_TEACHER_CHECKPOINT SGDRIVE_TEACHER_VLM_PATH SGDRIVE_TEACHER_VLM_TYPE
mkdir -p "${WORK_DIR}"

echo "[LatentSight][Launch] experiment=B3"
echo "[LatentSight][Launch] teacher=true future_queries=true distillation=true planner_condition=single"
echo "[LatentSight][Launch] train_mode=low_lr_full_finetune"
echo "[LatentSight][Launch] config=${CONFIG}"
echo "[LatentSight][Launch] student_checkpoint=${STUDENT_CHECKPOINT}"
echo "[LatentSight][Launch] teacher_checkpoint=${SGDRIVE_TEACHER_CHECKPOINT}"
echo "[LatentSight][Launch] work_dir=${WORK_DIR}"
echo "[LatentSight][Launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"

cd "${NAVSIM_DEVKIT_ROOT}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  navsim/planning/script/run_training_recogdrive.py \
  --config-path "$(dirname "${CONFIG}")" \
  --config-name "$(basename "${CONFIG}" .yaml)" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  output_dir="${WORK_DIR}" \
  experiment_name=latentsight_v2_b3_teacher_future_distill_single_condition \
  agent.checkpoint_path="${STUDENT_CHECKPOINT}" \
  agent.vlm_path="${STUDENT_VLM_PATH}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.num_nodes=1