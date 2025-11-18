#!/bin/bash

# ===== Proactive Agent GRPO Training with Group-Aware Rewards =====
# 4 GPU training with automatic data processing

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
export NCCL_IB_DISABLE=1  # 如果使用IB网络，可能需要禁用
export NCCL_DEBUG=WARN

# CUDA配置
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_LAUNCH_BLOCKING=0

# 尝试使用vLLM v0引擎（更稳定）
unset VLLM_USE_V1

# ===== 数据路径配置 =====
INPUT_DATA="data/sampleQA.jsonl"
OUTPUT_DATA_DIR="data/processed_sampleQA"
TRAIN_DATA="${OUTPUT_DATA_DIR}/train.parquet"
TEST_DATA="${OUTPUT_DATA_DIR}/test.parquet"

# ===== 模型路径配置 =====
# TODO: 修改为你的模型路径
MODEL_PATH="/mnt/hdd/Fangda/data/models/qwen3-8b"

# ===== 数据处理 =====
echo "============================================================================"
echo "Checking data status..."
echo "============================================================================"

if [ -f "${TRAIN_DATA}" ] && [ -f "${TEST_DATA}" ]; then
    echo "✓ Processed data found:"
    echo "  - ${TRAIN_DATA}"
    echo "  - ${TEST_DATA}"
    echo ""
else
    echo "✗ Processed data not found. Processing sampleQA.jsonl..."
    echo ""

    # Check if input data exists
    if [ ! -f "${INPUT_DATA}" ]; then
        echo "Error: Input data file not found: ${INPUT_DATA}"
        echo "Please ensure that sampleQA.jsonl exists in the data/ directory."
        exit 1
    fi

    echo "Processing data from ${INPUT_DATA}..."
    python3 examples/proactive/process_sampleQA.py \
        --input_file "${INPUT_DATA}" \
        --output_dir "${OUTPUT_DATA_DIR}" \
        --system_prompt "You are a helpful proactive assistant." \
        --split_ratio 0.95

    echo ""
    echo "✓ Data processing completed!"
    echo ""
fi

# ===== 开始训练 =====
echo "============================================================================"
echo "Starting Proactive Agent GRPO Training with Group-Aware Rewards..."
echo "============================================================================"
date

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
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TEST_DATA}" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
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
    actor_rollout_ref.rollout.n=16 \
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
    custom_reward_function.reward_kwargs.beta=0.5 \
    critic=null \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.log_val_generations=5 \
    trainer.project_name='proactive_agent' \
    trainer.experiment_name='sampleQA_grpo_beta0.5_n16_4gpu' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.default_local_dir='checkpoints/proactive_agent/sampleQA_grpo_beta0.5_n16_4gpu' \
    $@

echo ""
echo "============================================================================"
echo "Training completed!"
echo "============================================================================"
date
