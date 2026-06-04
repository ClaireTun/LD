#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LD}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/recogdrive}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-${REPO_ROOT}/work_dirs}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

CONFIG=${CONFIG:-${REPO_ROOT}/configs/latentsight/v2/b2_query_carrier_low_lr_full_finetune.yaml}
STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-/path/to/student_language_planner.ckpt}
STUDENT_VLM_PATH=${STUDENT_VLM_PATH:-/path/to/ReCogDrive-VLM-2B}
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/work_dirs/latentsight_v2/b2_query_carrier_low_lr_full_finetune}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29512}

export STUDENT_CHECKPOINT STUDENT_VLM_PATH
mkdir -p "${WORK_DIR}"

echo "[LatentSight][Launch] experiment=B2"
echo "[LatentSight][Launch] train_mode=low_lr_full_finetune"
echo "[LatentSight][Launch] teacher=false future_queries=true distillation=false"
echo "[LatentSight][Launch] config=${CONFIG}"
echo "[LatentSight][Launch] student_checkpoint=${STUDENT_CHECKPOINT}"
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
  experiment_name=latentsight_v2_b2_query_carrier_low_lr_full_finetune \
  agent.checkpoint_path="${STUDENT_CHECKPOINT}" \
  agent.vlm_path="${STUDENT_VLM_PATH}" \
  trainer.params.devices="${GPUS}" \
  trainer.params.num_nodes=1
