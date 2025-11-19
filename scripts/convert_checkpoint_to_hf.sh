#!/bin/bash

# ===== Checkpoint to HuggingFace Model Converter =====
# Converts FSDP checkpoints to HuggingFace format for inference
#
# Usage:
#   bash scripts/convert_checkpoint_to_hf.sh <checkpoint_dir> [output_dir]
#
# Example:
#   bash scripts/convert_checkpoint_to_hf.sh \
#       checkpoints/proactive_agent/sft50_rl_beta0.5_n16/epoch_10 \
#       models/proactive_agent_epoch10

set -e

# ===== 参数解析 =====
CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-models/converted_model}"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not provided!"
    echo ""
    echo "Usage:"
    echo "  bash scripts/convert_checkpoint_to_hf.sh <checkpoint_dir> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  # Convert RL checkpoint"
    echo "  bash scripts/convert_checkpoint_to_hf.sh \\"
    echo "      checkpoints/proactive_agent/sft50_rl_beta0.5_n16/epoch_10 \\"
    echo "      models/proactive_agent_epoch10"
    echo ""
    echo "  # Convert SFT checkpoint"
    echo "  bash scripts/convert_checkpoint_to_hf.sh \\"
    echo "      checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2 \\"
    echo "      models/proactive_sft"
    echo ""
    exit 1
fi

# ===== 检查 checkpoint 是否存在 =====
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# ===== 自动检测 checkpoint 类型 =====
echo "============================================================================"
echo "Checkpoint to HuggingFace Model Converter"
echo "============================================================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# 检测是否是 FSDP checkpoint（查找 model_world_size_*_rank_*.pt 文件）
if ls "$CHECKPOINT_DIR"/model_world_size_*_rank_*.pt 1> /dev/null 2>&1; then
    BACKEND="fsdp"
    echo "Detected: FSDP checkpoint"
# 检测是否是 Megatron checkpoint（查找 mp_rank_* 目录）
elif ls -d "$CHECKPOINT_DIR"/mp_rank_* 1> /dev/null 2>&1; then
    BACKEND="megatron"
    echo "Detected: Megatron checkpoint"
else
    echo "Error: Cannot determine checkpoint type!"
    echo "Expected FSDP files: model_world_size_*_rank_*.pt"
    echo "Or Megatron directories: mp_rank_*"
    exit 1
fi

echo ""

# ===== 运行转换 =====
echo "Starting conversion..."
echo ""

if [ "$BACKEND" = "fsdp" ]; then
    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "$CHECKPOINT_DIR" \
        --target_dir "$OUTPUT_DIR"
elif [ "$BACKEND" = "megatron" ]; then
    echo "Note: For Megatron checkpoints, you may need additional flags:"
    echo "  --tie-word-embedding (if your model ties word embeddings)"
    echo "  --is-value-model (if this is a value model)"
    echo ""
    python scripts/legacy_model_merger.py merge \
        --backend megatron \
        --local_dir "$CHECKPOINT_DIR" \
        --target_dir "$OUTPUT_DIR"
fi

# ===== 检查输出 =====
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/config.json" ]; then
    echo ""
    echo "============================================================================"
    echo "✓ Conversion successful!"
    echo "============================================================================"
    echo "HuggingFace model saved to: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR" | grep -E "\.(json|safetensors|bin|model)$" || ls -lh "$OUTPUT_DIR"
    echo ""
    echo "You can now use this model with:"
    echo "  - vLLM: vllm serve $OUTPUT_DIR"
    echo "  - Transformers: AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR')"
    echo "  - Any other HuggingFace-compatible tool"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "✗ Conversion may have failed"
    echo "============================================================================"
    echo "Output directory: $OUTPUT_DIR"
    echo "Please check the error messages above."
    exit 1
fi
