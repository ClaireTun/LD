export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/data/tuntun/project/navsim-main/dataset/maps"
export NAVSIM_EXP_ROOT="/home/data/tuntun/project/LSD/exp"
export NAVSIM_DEVKIT_ROOT="/home/data/tuntun/project/LSD/recogdrive"
export OPENSCENE_DATA_ROOT="/home/data/tuntun/project/navsim-main/dataset"
TRAIN_TEST_SPLIT=navtrain
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



torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=4 \
    --master_port=29509 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive.py \
    agent=B1 \
    agent.grpo=False \
    agent.vlm_path='/home/data/tuntun/project/SGDrive-main/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.cache_hidden_state=False \
    agent.cache_mode=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=10\
    trainer.params.num_nodes=1 \
    trainer.params.devices=4 \
    experiment_name=training_recogdrive_agent_B1 \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path='' \
    use_cache_without_dataset=False \
    force_cache_computation=False \
    deepspeed.bf16.enabled=false \
    deepspeed.fp16.enabled=false \
    trainer.params.precision=32-true