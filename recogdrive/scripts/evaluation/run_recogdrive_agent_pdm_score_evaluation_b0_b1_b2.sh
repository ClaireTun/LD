#!/usr/bin/env bash
set -euo pipefail
set -x

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"/path/to/NAVSIM/dataset/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/path/to/NAVSIM/exp"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/path/to/NAVSIM/navsim-main"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/path/to/NAVSIM/dataset"}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT PORT

echo "GPUS: ${GPUS}, GPUS_PER_NODE: ${GPUS_PER_NODE}, NODES: ${NODES}"

# Shared defaults. Override per agent with CHECKPOINT_B0/CHECKPOINT_B1/CHECKPOINT_B2.
CHECKPOINT=${CHECKPOINT:-"/path/to/recogdrive.ckpt"}
CHECKPOINT_B0=${CHECKPOINT_B0:-$CHECKPOINT}
CHECKPOINT_B1=${CHECKPOINT_B1:-$CHECKPOINT}
CHECKPOINT_B2=${CHECKPOINT_B2:-$CHECKPOINT}
VLM_PATH=${VLM_PATH:-"/path/to/ReCogDrive-VLM-2B"}

# Space-separated list; examples:
#   TEST_AGENTS="B0" sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_b0_b1_b2.sh
#   TEST_AGENTS="B1 B2" CHECKPOINT_B1=/path/b1.ckpt CHECKPOINT_B2=/path/b2.ckpt sh ...
TEST_AGENTS=${TEST_AGENTS:-"B0 B1 B2"}

checkpoint_for_agent() {
    case "$1" in
        B0) echo "$CHECKPOINT_B0" ;;
        B1) echo "$CHECKPOINT_B1" ;;
        B2) echo "$CHECKPOINT_B2" ;;
        *)
            echo "Unsupported agent '$1'. Expected one of: B0 B1 B2." >&2
            exit 2
            ;;
    esac
}

for AGENT_CONFIG in ${TEST_AGENTS}; do
    CKPT_PATH=$(checkpoint_for_agent "$AGENT_CONFIG")
    echo "===== Evaluating ${AGENT_CONFIG} with checkpoint ${CKPT_PATH} ====="

    # B0/B1/B2 yaml files define their own module topology. For PDM eval we only
    # override runtime paths and disable teacher/distillation-only branches.
    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py" \
        train_test_split="$TRAIN_TEST_SPLIT" \
        agent="$AGENT_CONFIG" \
        agent.checkpoint_path="$CKPT_PATH" \
        agent.vlm_path="$VLM_PATH" \
        agent.cache_hidden_state=False \
        agent.use_sgdrive_teacher=False \
        agent.use_structural_world_distill=False \
        experiment_name="recogdrive_agent_eval_${AGENT_CONFIG}"
done
