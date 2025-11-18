#!/bin/bash
# Run proactive agent GRPO training with group-aware rewards on 4 GPUs
#
# This script will:
# 1. Automatically process sampleQA.jsonl data if not already processed
# 2. Run GRPO training with group-aware rewards
#
# Usage:
#   bash examples/proactive/run_pro_grpo.sh
#
# Or with custom arguments:
#   bash examples/proactive/run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=0.7

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Set the number of GPUs (should match trainer.n_gpus_per_node in yaml)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Python path
PYTHON_EXEC=${PYTHON_EXEC:-python}

# Paths
INPUT_DATA="data/sampleQA.jsonl"
OUTPUT_DATA_DIR="data/processed_sampleQA"
TRAIN_DATA="${OUTPUT_DATA_DIR}/train.parquet"
TEST_DATA="${OUTPUT_DATA_DIR}/test.parquet"

# Config path
CONFIG_PATH="examples/proactive"
CONFIG_NAME="config_with_group_aware_reward"

# ============================================================================
# Step 1: Process data if needed
# ============================================================================

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
        echo ""
        echo "Please ensure that sampleQA.jsonl exists in the data/ directory."
        echo "Expected format:"
        echo '{"id": 0, "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...}'
        exit 1
    fi

    echo "Processing data from ${INPUT_DATA}..."
    $PYTHON_EXEC examples/proactive/process_sampleQA.py \
        --input_file "${INPUT_DATA}" \
        --output_dir "${OUTPUT_DATA_DIR}" \
        --system_prompt "You are a helpful proactive assistant." \
        --split_ratio 0.95

    echo ""
    echo "✓ Data processing completed!"
    echo ""
fi

# ============================================================================
# Step 2: Verify processed data
# ============================================================================

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: Training data not found at ${TRAIN_DATA}"
    echo "Data processing may have failed."
    exit 1
fi

if [ ! -f "${TEST_DATA}" ]; then
    echo "Warning: Test data not found at ${TEST_DATA}"
    echo "Training will proceed but validation may not work."
fi

# ============================================================================
# Step 3: Run Training
# ============================================================================

echo "============================================================================"
echo "Starting Proactive Agent GRPO Training"
echo "============================================================================"
echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Train data: ${TRAIN_DATA}"
echo "Test data: ${TEST_DATA}"
echo "============================================================================"
echo ""

$PYTHON_EXEC -m verl.trainer.main_ppo \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    "$@"  # Pass any additional arguments to the script

echo ""
echo "============================================================================"
echo "Training completed!"
echo "============================================================================"

# ============================================================================
# Useful command-line overrides
# ============================================================================
#
# Override beta value:
#   bash examples/proactive/run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=0.7
#
# Override number of rollouts per prompt:
#   bash examples/proactive/run_pro_grpo.sh ++actor_rollout_ref.rollout.n=8
#
# Override model path:
#   bash examples/proactive/run_pro_grpo.sh ++actor_rollout_ref.model.path=/path/to/model
#
# Change batch size:
#   bash examples/proactive/run_pro_grpo.sh ++data.train_batch_size=128
#
# Change number of epochs:
#   bash examples/proactive/run_pro_grpo.sh ++trainer.total_epochs=20
#
# Change experiment name:
#   bash examples/proactive/run_pro_grpo.sh ++trainer.experiment_name=my_experiment
#
# Use different reward function:
#   bash examples/proactive/run_pro_grpo.sh ++custom_reward_function.name=proactive_group_aware_reward_detailed
#
# Skip data processing (if you want to reprocess):
#   rm -rf data/processed_sampleQA
#   bash examples/proactive/run_pro_grpo.sh
#
# ============================================================================
