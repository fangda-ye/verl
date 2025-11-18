#!/bin/bash

# ===== Complete Training Pipeline: SFT Warmup + RL Training =====
# This script:
# 1. Randomly samples 50 items from sampleQA_processed_2.jsonl for SFT
# 2. Runs SFT warmup training
# 3. Automatically continues with RL training using the SFT checkpoint

# ===== 关键环境变量 - 必须在最开始设置 =====
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 禁用对称内存相关特性
export VLLM_USE_SYMMETRIC_MEMORY=0
export VLLM_SKIP_P2P_CHECK=1
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# NCCL配置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# CUDA配置
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_LAUNCH_BLOCKING=0

# 尝试使用vLLM v0引擎（更稳定）
unset VLLM_USE_V1

# ===== 配置 =====
# SFT 配置
SFT_INPUT="data/sampleQA_processed_2.jsonl"
SFT_OUTPUT="data/sft_samples.jsonl"
SFT_CHECKPOINT_DIR="checkpoints/proactive_sft_warmup"
NUM_SFT_SAMPLES=50
SFT_EPOCHS=2

# RL 配置
RL_INPUT="data/sampleQA.jsonl"
RL_OUTPUT_DIR="data/processed_sampleQA"
RL_TRAIN_DATA="${RL_OUTPUT_DIR}/train.parquet"
RL_TEST_DATA="${RL_OUTPUT_DIR}/test.parquet"

# 模型路径
BASE_MODEL_PATH="/mnt/hdd/Fangda/data/models/qwen3-8b"

# RL 训练参数
RL_BETA=0.5
RL_ROLLOUTS=16
RL_EPOCHS=10

echo "================================================================================"
echo "Complete Training Pipeline: SFT Warmup (${NUM_SFT_SAMPLES} samples) + RL Training"
echo "================================================================================"
echo "Stage 1: SFT Warmup"
echo "  Input: ${SFT_INPUT}"
echo "  Samples: ${NUM_SFT_SAMPLES}"
echo "  Epochs: ${SFT_EPOCHS}"
echo ""
echo "Stage 2: RL Training"
echo "  Input: ${RL_INPUT}"
echo "  Beta: ${RL_BETA}"
echo "  Rollouts per prompt: ${RL_ROLLOUTS}"
echo "  Epochs: ${RL_EPOCHS}"
echo "================================================================================"
echo ""

# ===== Stage 1: SFT Warmup =====
echo "================================================================================"
echo "Stage 1: SFT Warmup Training"
echo "================================================================================"
date
echo ""

# 处理 SFT 数据
if [ -f "${SFT_OUTPUT}" ]; then
    echo "✓ SFT data already exists: ${SFT_OUTPUT}"
else
    if [ ! -f "${SFT_INPUT}" ]; then
        echo "Error: SFT input file not found: ${SFT_INPUT}"
        exit 1
    fi

    echo "Processing SFT data..."
    python3 examples/proactive/process_sft_data.py \
        --input_file "${SFT_INPUT}" \
        --output_file "${SFT_OUTPUT}" \
        --num_samples ${NUM_SFT_SAMPLES} \
        --seed 42
    echo ""
fi

# 运行 SFT 训练
echo "Running SFT training..."
set -x

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_OUTPUT}" \
    data.val_files="${SFT_OUTPUT}" \
    data.micro_batch_size_per_gpu=4 \
    data.shuffle=True \
    model.partial_pretrain="${BASE_MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bfloat16 \
    optim.lr=1e-5 \
    optim.weight_decay=0.0 \
    trainer.default_local_dir="${SFT_CHECKPOINT_DIR}" \
    trainer.project_name=proactive_sft_warmup \
    trainer.experiment_name=sft_${NUM_SFT_SAMPLES}_samples \
    trainer.logger='["console","tensorboard"]' \
    trainer.total_epochs=${SFT_EPOCHS} \
    trainer.save_freq=1 \
    trainer.device=cuda \
    use_remove_padding=True

set +x

# 查找 SFT checkpoint
LATEST_SFT_CHECKPOINT=$(ls -td ${SFT_CHECKPOINT_DIR}/sft_${NUM_SFT_SAMPLES}_samples/checkpoints/epoch_* 2>/dev/null | head -1)

if [ -z "${LATEST_SFT_CHECKPOINT}" ]; then
    echo "Error: SFT checkpoint not found!"
    echo "Expected location: ${SFT_CHECKPOINT_DIR}/sft_${NUM_SFT_SAMPLES}_samples/checkpoints/"
    exit 1
fi

echo ""
echo "✓ SFT warmup completed!"
echo "  Checkpoint: ${LATEST_SFT_CHECKPOINT}"
echo ""

# ===== Stage 2: RL Training =====
echo "================================================================================"
echo "Stage 2: RL Training with Group-Aware Rewards"
echo "================================================================================"
date
echo ""

# 处理 RL 数据
if [ -f "${RL_TRAIN_DATA}" ] && [ -f "${RL_TEST_DATA}" ]; then
    echo "✓ RL data already processed"
else
    if [ ! -f "${RL_INPUT}" ]; then
        echo "Error: RL input file not found: ${RL_INPUT}"
        exit 1
    fi

    echo "Processing RL data..."
    python3 examples/proactive/process_sampleQA.py \
        --input_file "${RL_INPUT}" \
        --output_dir "${RL_OUTPUT_DIR}" \
        --system_prompt "You are a helpful proactive assistant." \
        --split_ratio 0.95
    echo ""
fi

# 运行 RL 训练（使用 SFT checkpoint）
echo "Running RL training from SFT checkpoint..."
echo "  Loading from: ${LATEST_SFT_CHECKPOINT}"
echo ""

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    data.train_files="${RL_TRAIN_DATA}" \
    data.val_files="${RL_TEST_DATA}" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${LATEST_SFT_CHECKPOINT}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.min_lr=1e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${RL_ROLLOUTS} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=group_aware \
    reward_model.num_examine=5 \
    custom_reward_function.path=examples/proactive/group_aware_reward.py \
    custom_reward_function.name=proactive_group_aware_reward \
    custom_reward_function.reward_kwargs.beta=${RL_BETA} \
    critic=null \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.log_val_generations=5 \
    trainer.project_name='proactive_agent' \
    trainer.experiment_name="sft${NUM_SFT_SAMPLES}_rl_beta${RL_BETA}_n${RL_ROLLOUTS}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${RL_EPOCHS} \
    trainer.default_local_dir="checkpoints/proactive_agent/sft${NUM_SFT_SAMPLES}_rl_beta${RL_BETA}_n${RL_ROLLOUTS}" \
    $@

set +x

echo ""
echo "================================================================================"
echo "Complete pipeline finished!"
echo "================================================================================"
echo "SFT checkpoint: ${LATEST_SFT_CHECKPOINT}"
echo "RL checkpoints: checkpoints/proactive_agent/sft${NUM_SFT_SAMPLES}_rl_beta${RL_BETA}_n${RL_ROLLOUTS}/"
date
