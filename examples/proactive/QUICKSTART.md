# Quick Start Guide - Proactive Agent Training

## 一键启动训练

### 前提条件

1. **数据准备**：确保 `data/sampleQA.jsonl` 存在
2. **模型路径**：在配置中设置你的模型路径

### 最简单的启动方式

```bash
# 从项目根目录运行
bash examples/proactive/run_pro_grpo.sh
```

就是这么简单！脚本会自动：

1. ✅ 检查数据是否已处理
2. ✅ 如果未处理，自动转换 `data/sampleQA.jsonl`
3. ✅ 启动 4-GPU GRPO 训练

## 详细步骤

### 步骤 1：准备数据

将你的 `sampleQA.jsonl` 文件放在 `data/` 目录：

```bash
# 数据应该在这个位置
data/sampleQA.jsonl
```

**数据格式示例**：

```json
{"id": 0, "messages": [{"role": "user", "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"}, {"role": "assistant", "content": "Michio Sugeno"}], "answer": {...}, "sub_category": "simpleQA"}
```

### 步骤 2：配置模型路径

编辑 `examples/proactive/config_with_group_aware_reward.yaml`：

```yaml
actor_rollout_ref:
  model:
    path: ~/models/Qwen2.5-7B-Instruct  # 改为你的模型路径
```

### 步骤 3：运行训练

```bash
bash examples/proactive/run_pro_grpo.sh
```

## 输出说明

### 数据处理输出

首次运行时，你会看到：

```
============================================================================
Checking data status...
============================================================================
✗ Processed data not found. Processing sampleQA.jsonl...

Processing sampleQA data from data/sampleQA.jsonl...
Loaded 1000 items
Successfully converted 1000 items

Splitting data:
  Train: 950 items (95.0%)
  Test: 50 items (5.0%)

Saved training data to data/processed_sampleQA/train.parquet
Saved example to data/processed_sampleQA/train_example.json
Saved test data to data/processed_sampleQA/test.parquet

✓ Data processing completed!
```

### 训练启动输出

```
============================================================================
Starting Proactive Agent GRPO Training
============================================================================
Config: examples/proactive/config_with_group_aware_reward.yaml
GPUs: 0,1,2,3
Train data: data/processed_sampleQA/train.parquet
Test data: data/processed_sampleQA/test.parquet
============================================================================
```

## 处理后的数据格式

查看 `data/processed_sampleQA/train_example.json` 可以看到处理后的数据：

```json
{
  "data_source": "sampleQA",
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful proactive assistant."
    },
    {
      "role": "user",
      "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"
    }
  ],
  "ability": "reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": "Michio Sugeno"
  },
  "extra_info": {
    "index": 0,
    "original_id": 0,
    "sub_category": "simpleQA"
  }
}
```

注意：
- ✅ 添加了系统提示词："You are a helpful proactive assistant."
- ✅ 只使用 `messages` 中的 user 和 assistant 内容
- ✅ `answer` 字段被忽略（仅用于 GT）

## 常用命令

### 调整训练参数

```bash
# 修改 beta 值（proactive 奖励权重）
bash examples/proactive/run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.7

# 修改 rollout 数量
bash examples/proactive/run_pro_grpo.sh \
    ++actor_rollout_ref.rollout.n=8

# 修改 batch size（如果 OOM）
bash examples/proactive/run_pro_grpo.sh \
    ++data.train_batch_size=128 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# 修改训练轮数
bash examples/proactive/run_pro_grpo.sh \
    ++trainer.total_epochs=20

# 组合多个参数
bash examples/proactive/run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.6 \
    ++actor_rollout_ref.rollout.n=12 \
    ++trainer.total_epochs=15
```

### 重新处理数据

如果你更新了 `sampleQA.jsonl`，需要重新处理：

```bash
# 删除旧的处理结果
rm -rf data/processed_sampleQA

# 重新运行（会自动处理）
bash examples/proactive/run_pro_grpo.sh
```

### 手动处理数据

如果你想单独处理数据：

```bash
python examples/proactive/process_sampleQA.py \
    --input_file data/sampleQA.jsonl \
    --output_dir data/processed_sampleQA \
    --split_ratio 0.95
```

## GPU 配置

### 修改 GPU 数量

**使用 2 GPU**：

1. 编辑 `config_with_group_aware_reward.yaml`：
   ```yaml
   trainer:
     n_gpus_per_node: 2
   ```

2. 编辑 `run_pro_grpo.sh`：
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   ```

**使用 8 GPU**：

1. 编辑配置：`n_gpus_per_node: 8`
2. 编辑脚本：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

### 指定特定 GPU

```bash
# 使用 GPU 2 和 3
export CUDA_VISIBLE_DEVICES=2,3
bash examples/proactive/run_pro_grpo.sh
```

## 验证数据处理

处理完成后，检查：

```bash
# 查看处理后的文件
ls -lh data/processed_sampleQA/
# 应该看到：
# train.parquet
# test.parquet
# train_example.json

# 查看示例数据
cat data/processed_sampleQA/train_example.json | jq .
```

## 故障排除

### 问题：找不到 sampleQA.jsonl

```
Error: Input data file not found: data/sampleQA.jsonl
```

**解决方案**：确保数据文件存在于正确位置：

```bash
ls -l data/sampleQA.jsonl
```

### 问题：ModuleNotFoundError

```
ModuleNotFoundError: No module named 'datasets'
```

**解决方案**：安装必要的依赖：

```bash
pip install datasets pyarrow
```

### 问题：CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决方案**：减小 batch size：

```bash
bash examples/proactive/run_pro_grpo.sh \
    ++data.train_batch_size=128 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    ++actor_rollout_ref.rollout.n=8
```

### 问题：模型加载失败

```
Error loading model from path...
```

**解决方案**：检查并更新配置中的模型路径：

```yaml
actor_rollout_ref:
  model:
    path: /correct/path/to/your/model
```

## 监控训练

### Tensorboard

训练过程中会自动记录到 tensorboard：

```bash
tensorboard --logdir checkpoints/proactive_agent/
```

### 检查 checkpoints

```bash
ls -l checkpoints/proactive_agent/grpo_beta0.5_n16/
```

## 下一步

训练完成后，你可以：

1. **评估模型**：使用测试集评估
2. **调整超参数**：尝试不同的 beta 值
3. **分析结果**：查看 tensorboard 日志
4. **部署模型**：使用训练好的 checkpoint

## 完整的工作流示例

```bash
# 1. 准备数据
cp /path/to/your/sampleQA.jsonl data/

# 2. （可选）查看数据格式
head -n 1 data/sampleQA.jsonl | jq .

# 3. 运行训练（自动处理数据）
bash examples/proactive/run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.5 \
    ++actor_rollout_ref.rollout.n=16 \
    ++trainer.total_epochs=10

# 4. 监控训练
tensorboard --logdir checkpoints/proactive_agent/

# 5. 训练完成后查看结果
ls -l checkpoints/proactive_agent/grpo_beta0.5_n16/
```

## 总结

使用这个工作流，你只需要：

1. ✅ 准备 `data/sampleQA.jsonl`
2. ✅ 配置模型路径
3. ✅ 运行 `bash examples/proactive/run_pro_grpo.sh`

其他一切都是自动的！🚀
