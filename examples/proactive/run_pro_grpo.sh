#!/bin/bash
# Run proactive agent GRPO training with group-aware rewards on 4 GPUs
#
# Usage:
#   bash examples/proactive/run_pro_grpo.sh
#
# Make sure to:
# 1. Update the data paths in config_with_group_aware_reward.yaml
# 2. Update the model path in config_with_group_aware_reward.yaml
# 3. Convert your JSONL data using convert_data.py first

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Set the number of GPUs (should match trainer.n_gpus_per_node in yaml)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Python path
PYTHON_EXEC=${PYTHON_EXEC:-python}

# Config path
CONFIG_PATH="examples/proactive"
CONFIG_NAME="config_with_group_aware_reward"

# ============================================================================
# Optional: Override config values via command line
# ============================================================================

# You can override any config value by passing it as an argument
# For example:
#   bash run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=0.7

# ============================================================================
# Data Conversion (if needed)
# ============================================================================

# Uncomment and modify if you need to convert your JSONL data first:
# echo "Converting JSONL data to veRL format..."
# $PYTHON_EXEC examples/proactive/convert_data.py \
#     --input_file /path/to/your/data.jsonl \
#     --output_dir ~/data/proactive_dataset \
#     --data_source_name proactive_dataset \
#     --ability reasoning \
#     --question_field question \
#     --answer_field answer

# ============================================================================
# Run Training
# ============================================================================

echo "============================================================================"
echo "Starting Proactive Agent GRPO Training"
echo "============================================================================"
echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================================"

$PYTHON_EXEC -m verl.trainer.main_ppo \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    "$@"  # Pass any additional arguments to the script

echo "============================================================================"
echo "Training completed!"
echo "============================================================================"

# ============================================================================
# Useful command-line overrides
# ============================================================================
#
# Override beta value:
#   bash run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=0.7
#
# Override number of rollouts per prompt:
#   bash run_pro_grpo.sh ++actor_rollout_ref.rollout.n=8
#
# Override model path:
#   bash run_pro_grpo.sh ++actor_rollout_ref.model.path=/path/to/model
#
# Override data path:
#   bash run_pro_grpo.sh ++data.train_files=/path/to/train.parquet
#
# Change batch size:
#   bash run_pro_grpo.sh ++data.train_batch_size=128
#
# Change number of epochs:
#   bash run_pro_grpo.sh ++trainer.total_epochs=20
#
# Disable wandb logging:
#   bash run_pro_grpo.sh trainer.logger='[console, tensorboard]'
#
# Change experiment name:
#   bash run_pro_grpo.sh ++trainer.experiment_name=my_experiment
#
# Use different reward function:
#   bash run_pro_grpo.sh ++custom_reward_function.name=proactive_group_aware_reward_detailed
#
# ============================================================================
