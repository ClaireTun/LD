set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1


CHECKPOINT="/path/to/recogdrive.ckpt"


VLM_PATH="/path/to/ReCogDrive-VLM-2B"

# B1 planner-conditioning settings. These must match the checkpoint used for
# evaluation; otherwise student_world_adapter/world_condition_projector weights
# will be filtered out during checkpoint loading and H_future will not condition
# planner inference. Teacher/distillation modules stay disabled for PDM eval.
STUDENT_WORLD_DIM=${STUDENT_WORLD_DIM:-256}
STUDENT_WORLD_NUM_TOKENS=${STUDENT_WORLD_NUM_TOKENS:-1}
STUDENT_WORLD_USE_MLP=${STUDENT_WORLD_USE_MLP:-True}
STUDENT_WORLD_USE_CROSS_ATTN=${STUDENT_WORLD_USE_CROSS_ATTN:-False}
STUDENT_WORLD_DROPOUT=${STUDENT_WORLD_DROPOUT:-0.0}
WORLD_CONDITION_FUSION_TYPE=${WORLD_CONDITION_FUSION_TYPE:-concat}
WORLD_CONDITION_DETACH=${WORLD_CONDITION_DETACH:-False}
WORLD_CONDITION_WEIGHT=${WORLD_CONDITION_WEIGHT:-1.0}


torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.vlm_path="$VLM_PATH" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.use_sgdrive_teacher=False \
    agent.teacher_feature_source="cached" \
    agent.use_student_world_adapter=True \
    agent.student_world_dim=${STUDENT_WORLD_DIM} \
    agent.student_world_num_tokens=${STUDENT_WORLD_NUM_TOKENS} \
    agent.student_world_use_mlp=${STUDENT_WORLD_USE_MLP} \
    agent.student_world_use_cross_attn=${STUDENT_WORLD_USE_CROSS_ATTN} \
    agent.student_world_dropout=${STUDENT_WORLD_DROPOUT} \
    agent.use_structural_world_distill=False \
    agent.use_world_tokens_as_planner_condition=True \
    agent.world_condition_fusion_type=${WORLD_CONDITION_FUSION_TYPE} \
    agent.world_condition_detach=${WORLD_CONDITION_DETACH} \
    agent.world_condition_weight=${WORLD_CONDITION_WEIGHT} \
    agent.use_world_denoise_influence=False \
    experiment_name=recogdrive_agent_eval_b1_world_condition

