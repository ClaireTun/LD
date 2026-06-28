#!/usr/bin/env bash
set -euo pipefail

# Two-stage reactive / pseudo closed-loop PDM evaluation for a DriveMem
# ReCogDrive student checkpoint. DriveMem and student-world modules are inferred
# from checkpoint tensors by agent.auto_configure_from_checkpoint=true.
# Override these variables from the shell:
#   CHECKPOINT=/path/to/drivemem_student.ckpt
#   VLM_PATH=/path/to/ReCogDrive-VLM-2B
#   CACHE_PATH=/path/to/metric_cache

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navhard_two_stage}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the DriveMem student checkpoint path}
VLM_PATH=${VLM_PATH:?Set VLM_PATH to the ReCogDrive VLM directory}
CACHE_PATH=${CACHE_PATH:?Set CACHE_PATH to the NAVSIM metric cache path}
SYNTHETIC_SENSOR_PATH=${SYNTHETIC_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/sensor_blobs}
SYNTHETIC_SCENES_PATH=${SYNTHETIC_SCENES_PATH:-$OPENSCENE_DATA_ROOT/${TRAIN_TEST_SPLIT}/synthetic_scene_pickles}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-drivemem_pdm_score}
DIT_TYPE=${DIT_TYPE:-small}
VLM_SIZE=${VLM_SIZE:-small}
VLM_TYPE=${VLM_TYPE:-internvl}

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  agent=recogdrive_agent \
  worker=single_machine_thread_pool \
  agent.checkpoint_path="$CHECKPOINT" \
  agent.vlm_path="$VLM_PATH" \
  agent.vlm_type="$VLM_TYPE" \
  agent.dit_type="$DIT_TYPE" \
  agent.vlm_size="$VLM_SIZE" \
  agent.cache_hidden_state=true \
  agent.cache_mode=false \
  agent.auto_configure_from_checkpoint=true \
  agent.drivemem.enabled=true \
  experiment_name="$EXPERIMENT_NAME" \
  metric_cache_path="$CACHE_PATH" \
  synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH" \
  synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"
