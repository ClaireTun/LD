#!/usr/bin/env bash
set -euo pipefail

# Two-stage reactive / pseudo closed-loop PDM evaluation for a legacy
# ReCogDrive baseline checkpoint. Override these variables from the shell:
#   CHECKPOINT=/path/to/ReCogDrive_Diffusion_Planner_*.ckpt
#   VLM_PATH=/path/to/ReCogDrive-VLM-2B
#   CACHE_PATH=/path/to/metric_cache

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navhard_two_stage}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the ReCogDrive planner checkpoint path}
VLM_PATH=${VLM_PATH:?Set VLM_PATH to the ReCogDrive VLM directory}
CACHE_PATH=${CACHE_PATH:?Set CACHE_PATH to the NAVSIM metric cache path}
SYNTHETIC_SENSOR_PATH=${SYNTHETIC_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs}
SYNTHETIC_SCENES_PATH=${SYNTHETIC_SCENES_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recogdrive_baseline_pdm_score}
DIT_TYPE=${DIT_TYPE:-small}
VLM_SIZE=${VLM_SIZE:-small}
VLM_TYPE=${VLM_TYPE:-internvl}

NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export NAVSIM_DEVKIT_ROOT

# Hydra default_evaluation.yaml resolves output_dir from NAVSIM_EXP_ROOT.
# Provide a local default so the script works when the caller has not exported it.
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}
export NAVSIM_EXP_ROOT

if [[ -n "${OPENSCENE_DATA_ROOT:-}" ]]; then
  NAVSIM_LOG_PATH=${NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/$DATA_SPLIT}
  ORIGINAL_SENSOR_PATH=${ORIGINAL_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/sensor_blobs/$DATA_SPLIT}
  SYNTHETIC_SENSOR_PATH=${SYNTHETIC_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs}
  SYNTHETIC_SCENES_PATH=${SYNTHETIC_SCENES_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles}
else
  : "${NAVSIM_LOG_PATH:?Set OPENSCENE_DATA_ROOT or NAVSIM_LOG_PATH}"
  : "${ORIGINAL_SENSOR_PATH:?Set OPENSCENE_DATA_ROOT or ORIGINAL_SENSOR_PATH}"
  : "${SYNTHETIC_SENSOR_PATH:?Set OPENSCENE_DATA_ROOT or SYNTHETIC_SENSOR_PATH}"
  : "${SYNTHETIC_SCENES_PATH:?Set OPENSCENE_DATA_ROOT or SYNTHETIC_SCENES_PATH}"
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recogdrive_baseline_pdm_score}
DIT_TYPE=${DIT_TYPE:-small}
VLM_SIZE=${VLM_SIZE:-small}
VLM_TYPE=${VLM_TYPE:-internvl}

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  hydra.searchpath="[file://$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common,pkg://navsim.planning.script.config.common]" \
  agent=recogdrive_agent \
  worker=single_machine_thread_pool \
  agent.checkpoint_path="$CHECKPOINT" \
  agent.vlm_path="$VLM_PATH" \
  agent.vlm_type="$VLM_TYPE" \
  agent.dit_type="$DIT_TYPE" \
  agent.vlm_size="$VLM_SIZE" \
  agent.cache_hidden_state=true \
  agent.cache_mode=false \
  agent.load_lidar=false \
  agent.auto_configure_from_checkpoint=true \
  experiment_name="$EXPERIMENT_NAME" \
  metric_cache_path="$CACHE_PATH" \
  navsim_log_path="$NAVSIM_LOG_PATH" \
  original_sensor_path="$ORIGINAL_SENSOR_PATH" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"
