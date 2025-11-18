#!/bin/bash

# ===== SFT Warmup for Proactive Agent =====
# Supervised fine-tuning with 50 random samples before RL training

# ===== 关键环境变量 =====
export CUDA_VISIBLE_DEVICES=0,1,2,3

# CUDA配置
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_LAUNCH_BLOCKING=0

# ===== 配置 =====
SFT_INPUT="data/sampleQA_processed_2.jsonl"
SFT_OUTPUT="data/sft_samples.jsonl"
SFT_CHECKPOINT_DIR="checkpoints/proactive_sft_warmup"
NUM_SFT_SAMPLES=50
SFT_EPOCHS=2

# 模型路径
MODEL_PATH="/mnt/hdd/Fangda/data/models/qwen3-8b"

# ===== 步骤 1: 处理 SFT 数据 =====
echo "============================================================================"
echo "Step 1: Processing SFT data..."
echo "============================================================================"

if [ -f "${SFT_OUTPUT}" ]; then
    echo "✓ SFT data already exists: ${SFT_OUTPUT}"
    echo "  To regenerate, delete the file and run again."
    echo ""
else
    if [ ! -f "${SFT_INPUT}" ]; then
        echo "Error: SFT input file not found: ${SFT_INPUT}"
        echo "Please ensure that sampleQA_processed_2.jsonl exists."
        exit 1
    fi

    python3 examples/proactive/process_sft_data.py \
        --input_file "${SFT_INPUT}" \
        --output_file "${SFT_OUTPUT}" \
        --num_samples ${NUM_SFT_SAMPLES} \
        --seed 42

    echo ""
    echo "✓ SFT data processing completed!"
    echo ""
fi

# ===== 步骤 2: 运行 SFT 训练 =====
echo "============================================================================"
echo "Step 2: Running SFT warmup training..."
echo "============================================================================"
date

set -x

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_OUTPUT}" \
    data.val_files="${SFT_OUTPUT}" \
    data.micro_batch_size_per_gpu=4 \
    data.shuffle=True \
    model.partial_pretrain="${MODEL_PATH}" \
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
    use_remove_padding=True \
    $@

set +x

echo ""
echo "============================================================================"
echo "SFT warmup completed!"
echo "============================================================================"
echo "Checkpoint saved to: ${SFT_CHECKPOINT_DIR}"
date

# 显示 checkpoint 路径
LATEST_CHECKPOINT=$(ls -td ${SFT_CHECKPOINT_DIR}/sft_${NUM_SFT_SAMPLES}_samples/checkpoints/epoch_* 2>/dev/null | head -1)
if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo ""
    echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
    echo ""
    echo "To continue with RL training, use:"
    echo "  bash examples/proactive/run_proactive_grpo.sh \\"
    echo "    actor_rollout_ref.model.path='${LATEST_CHECKPOINT}'"
fi
