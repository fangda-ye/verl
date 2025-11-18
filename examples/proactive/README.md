# Proactive Agent Training with Group-Aware Rewards

本目录包含使用 `GroupAwareRewardManager` 训练 proactive agent 的完整示例，支持 4 GPU 训练。

## 什么是 Proactive Agent？

Proactive agent 是一种能够主动思考和推理的智能体，其回答分为三个部分：

1. **`<think>...</think>`** - 思考过程
2. **`<proactive>...</proactive>`** - 主动推理/预测
3. **正式回答** - 最终答案（在标签之外的内容）

## 核心特性

### 1. **Group-Aware Reward Scoring**

- 在计算奖励时，可以看到同一个 prompt 的所有 rollout responses
- 支持基于 group 统计信息的自适应奖励策略
- 详见 `group_aware_reward.py` 中的实现

### 2. **改进的正确性判断**

- `is_correct` 只判断**正式回答**部分（移除 `<think>` 和 `<proactive>` 标签）
- 避免将思考过程中的内容误判为最终答案

### 3. **Proactive 奖励策略**

根据 response 的类型给予不同奖励：

- **有 `<proactive>` 内容**：
  ```
  reward = beta * (1 - group_acc) + format_bonus
  ```
  难题（低 group_acc）获得更高奖励

- **无 `<proactive>` 内容**：
  ```
  reward = correctness + format_bonus
  ```
  基于正确性的标准奖励

- **格式奖励**：
  - `<think>` 标签：+0.05
  - `<proactive>` 标签：+0.05

## 文件说明

```
examples/proactive/
├── README.md                           # 本文档
├── process_sampleQA.py                # sampleQA 数据处理脚本
├── convert_data.py                     # 通用数据转换脚本
├── group_aware_reward.py              # 奖励函数实现
├── config_with_group_aware_reward.yaml # 训练配置（4 GPU）
└── run_pro_grpo.sh                    # 训练脚本（自动处理数据）
```

## 快速开始（使用 sampleQA 数据）

**最简单的方式 - 一键启动**：

1. 确保 `data/sampleQA.jsonl` 存在
2. 运行：
   ```bash
   bash examples/proactive/run_pro_grpo.sh
   ```

脚本会自动：
- 检查数据是否已处理
- 如果未处理，自动运行 `process_sampleQA.py` 转换数据
- 启动 GRPO 训练

**数据格式说明**：

`data/sampleQA.jsonl` 的每一行应该是：

```json
{
  "id": 0,
  "messages": [
    {"role": "user", "content": "Who received the IEEE Frank Rosenblatt Award in 2010?"},
    {"role": "assistant", "content": "Michio Sugeno"}
  ],
  "answer": {...},
  "sub_category": "simpleQA"
}
```

处理脚本会：
- 提取 `messages[0]` (user) 作为 prompt
- 提取 `messages[1]` (assistant) 作为 ground truth
- 添加系统提示词："You are a helpful proactive assistant."
- 自动分割为 train/test (95%/5%)
- 保存到 `data/processed_sampleQA/`

## 快速开始（使用自定义数据）

### 步骤 1：准备数据

#### 1.1 如果你的数据是 JSONL 格式

假设你的数据格式为：

```json
{"question": "What is 2+2?", "answer": "4"}
{"question": "What is the capital of France?", "answer": "Paris"}
```

使用 `convert_data.py` 转换：

```bash
python examples/proactive/convert_data.py \
    --input_file /path/to/your/data.jsonl \
    --output_dir ~/data/proactive_dataset \
    --data_source_name proactive_dataset \
    --ability reasoning \
    --question_field question \
    --answer_field answer
```

**自定义参数**：

- `--system_prompt`: 可选的系统提示词
- `--question_field`: JSONL 中问题的字段名（默认：`question`）
- `--answer_field`: JSONL 中答案的字段名（默认：`answer`）

#### 1.2 如果你的数据已经是 veRL 格式

如果数据已经是 parquet 格式，直接跳到步骤 2。

### 步骤 2：配置训练参数

编辑 `config_with_group_aware_reward.yaml`：

```yaml
# 1. 更新数据路径
data:
  train_files: ~/data/proactive_dataset/train.parquet
  val_files: ~/data/proactive_dataset/test.parquet  # 可选

# 2. 更新模型路径
actor_rollout_ref:
  model:
    path: ~/models/Qwen2.5-7B-Instruct  # 你的模型路径

# 3. 调整 GRPO 参数
actor_rollout_ref:
  rollout:
    n: 16  # 每个 prompt 生成多少个 responses

# 4. 调整奖励函数参数
custom_reward_function:
  reward_kwargs:
    beta: 0.5  # Proactive 奖励权重 (0.0-1.0)
```

### 步骤 3：运行训练

```bash
# 基础运行
bash examples/proactive/run_pro_grpo.sh

# 或者通过命令行覆盖参数
bash examples/proactive/run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.7 \
    ++actor_rollout_ref.rollout.n=8 \
    ++trainer.total_epochs=20
```

## 配置详解

### GPU 配置

默认配置为 **4 GPU 训练**：

```yaml
trainer:
  nnodes: 1            # 单节点
  n_gpus_per_node: 4   # 4 张 GPU
```

在 `run_pro_grpo.sh` 中：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**修改 GPU 数量**：

- 2 GPU: 修改为 `n_gpus_per_node: 2` 和 `CUDA_VISIBLE_DEVICES=0,1`
- 8 GPU: 修改为 `n_gpus_per_node: 8` 和 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

### 批次大小配置

根据你的 GPU 显存调整：

```yaml
data:
  train_batch_size: 256  # 全局批次大小

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 64  # PPO mini-batch size
    ppo_micro_batch_size_per_gpu: 16  # 每张 GPU 的 micro-batch size
```

**显存估算**：

- `ppo_micro_batch_size_per_gpu * n_gpus_per_node` 应该能被 `ppo_mini_batch_size` 整除
- 增大 `ppo_micro_batch_size_per_gpu` 会增加显存使用
- 如果 OOM，减小 `ppo_micro_batch_size_per_gpu` 或 `train_batch_size`

### GRPO 参数

```yaml
algorithm:
  adv_estimator: grpo  # 必须是 grpo
  norm_adv_by_std_in_grpo: true  # true: GRPO, false: Dr.GRPO

actor_rollout_ref:
  rollout:
    n: 16  # 每个 prompt 的 rollout 数量，建议 4-16
```

### Reward Function 参数

```yaml
custom_reward_function:
  name: proactive_group_aware_reward  # 奖励函数名称
  reward_kwargs:
    beta: 0.5  # Proactive 奖励权重
```

**可选的奖励函数**：

- `proactive_group_aware_reward` - 主要函数，返回简单分数
- `proactive_group_aware_reward_detailed` - 返回详细信息（用于调试）
- `simple_group_aware_reward` - 最简单的示例
- `adaptive_difficulty_reward` - 基于 group 难度的自适应奖励
- `best_of_n_reward` - Best-of-N 策略

## 数据格式

### 输入 JSONL 格式

```json
{
  "question": "What is 2+2?",
  "answer": "4"
}
```

### 转换后的 veRL 格式

```json
{
  "data_source": "proactive_dataset",
  "prompt": [{"role": "user", "content": "What is 2+2?"}],
  "ability": "reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": "4"
  },
  "extra_info": {
    "index": 0
  }
}
```

## Response 格式

训练过程中，模型生成的 response 应该包含以下部分（可选）：

```
<think>
Let me think about this problem...
2 + 2 is a simple addition.
</think>

<proactive>
I should also mention that addition is commutative.
</proactive>

The answer is 4.
```

- **`<think>...</think>`**: 思考过程（可选）
- **`<proactive>...</proactive>`**: 主动推理（可选）
- **正式回答**: 在标签外的内容

**注意**：

- `is_correct` 只判断正式回答部分（"The answer is 4."）
- 思考和 proactive 部分不参与正确性判断，但会获得格式奖励

## 奖励函数工作原理

### 提取正式回答

```python
def extract_formal_answer(response: str) -> str:
    # 移除 <think>...</think>
    formal_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # 移除 <proactive>...</proactive>
    formal_answer = re.sub(r'<proactive>.*?</proactive>', '', formal_answer, flags=re.DOTALL)
    return formal_answer.strip()
```

### 计算奖励

```python
# 1. 判断正确性（只看正式回答）
formal_answer = extract_formal_answer(response)
is_correct = ground_truth in formal_answer

# 2. 计算 group 准确率
group_acc = sum(correctness_scores) / len(correctness_scores)

# 3. 计算格式奖励
format_bonus = 0.0
if has_think:
    format_bonus += 0.05
if has_proactive:
    format_bonus += 0.05

# 4. 计算最终奖励
if has_proactive:
    # Proactive 回答：难题获得更高奖励
    reward = beta * (1 - group_acc) + format_bonus
else:
    # 普通回答：基于正确性
    reward = correctness + format_bonus
```

## 调试和监控

### 打印 Group 信息

在配置中设置：

```yaml
reward_model:
  num_examine: 5  # 打印前 5 个 groups
```

输出示例：

```
[Group UID: abc123]
[prompt] What is 2+2?
[num_responses] 16

  [response_0] <think>Simple addition</think> The answer is 4.
  [ground_truth_0] 4
  [score_0] 1.1  # 1.0 (correct) + 0.05 (think) + 0.05 (no proactive)

  [response_1] <proactive>Related concepts...</proactive> The answer is 5.
  [ground_truth_1] 4
  [score_1] 0.55  # beta * (1 - group_acc) + 0.05
  ...
```

### 使用详细奖励函数

修改配置：

```yaml
custom_reward_function:
  name: proactive_group_aware_reward_detailed
```

会返回额外信息：

```python
{
    "score": 1.05,
    "correctness": 1.0,
    "has_think": True,
    "has_proactive": False,
    "format_bonus": 0.05,
    "group_acc": 0.75,
    "formal_answer": "The answer is 4."
}
```

## 常见命令行覆盖

```bash
# 更改 beta 值
bash run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=0.7

# 更改 rollout 数量
bash run_pro_grpo.sh ++actor_rollout_ref.rollout.n=8

# 更改模型路径
bash run_pro_grpo.sh ++actor_rollout_ref.model.path=/path/to/model

# 更改数据路径
bash run_pro_grpo.sh ++data.train_files=/path/to/train.parquet

# 更改批次大小
bash run_pro_grpo.sh ++data.train_batch_size=128

# 更改训练轮数
bash run_pro_grpo.sh ++trainer.total_epochs=20

# 组合多个参数
bash run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.6 \
    ++actor_rollout_ref.rollout.n=12 \
    ++trainer.total_epochs=15
```

## 常见问题

### Q: 数据文件在哪里？

A: 你需要先运行 `convert_data.py` 将你的 JSONL 数据转换为 parquet 格式。转换后的文件会保存在 `--output_dir` 指定的目录。

### Q: 如何修改 GPU 数量？

A:
1. 修改 yaml 中的 `trainer.n_gpus_per_node`
2. 修改 sh 脚本中的 `CUDA_VISIBLE_DEVICES`

### Q: OOM 怎么办？

A: 减小以下参数：
- `data.train_batch_size`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.n`

### Q: Beta 参数如何选择？

A:
- `beta=0.0`: 不鼓励 proactive 行为
- `beta=0.5`: 平衡策略（推荐）
- `beta=1.0`: 强烈鼓励 proactive 行为
- 建议从 0.5 开始，根据效果调整

### Q: 如何验证数据转换是否正确？

A: 查看输出目录中的 `train_example.json`，检查格式是否正确。

### Q: 能否使用其他奖励函数？

A: 可以！在 `group_aware_reward.py` 中编写自己的函数，然后在配置中指定函数名。

## 实验建议

### 超参数搜索

尝试不同的 `beta` 值：

```bash
for beta in 0.3 0.5 0.7; do
    bash run_pro_grpo.sh ++custom_reward_function.reward_kwargs.beta=$beta
done
```

### A/B 测试

对比有无 proactive 奖励的效果：

```bash
# 无 proactive 奖励
bash run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.0 \
    ++trainer.experiment_name=no_proactive

# 有 proactive 奖励
bash run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.5 \
    ++trainer.experiment_name=with_proactive
```

## 进阶用法

### 自定义奖励函数

编辑 `group_aware_reward.py`，添加你自己的函数：

```python
def my_custom_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    # 你的逻辑
    ...
    return scores
```

然后在配置中使用：

```yaml
custom_reward_function:
  name: my_custom_reward
```

### 多阶段训练

```bash
# 第一阶段：鼓励格式
bash run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.3 \
    ++trainer.total_epochs=5

# 第二阶段：强化 proactive
bash run_pro_grpo.sh \
    ++custom_reward_function.reward_kwargs.beta=0.7 \
    ++trainer.total_epochs=10 \
    ++trainer.resume_from_path=checkpoints/proactive_agent/grpo_beta0.3_n16/epoch_5
```

## 参考资料

- [GroupAwareRewardManager 文档](../../docs/workers/group_aware_reward_manager.md)
- [GRPO 论文](https://arxiv.org/abs/2402.03300)
- [veRL 文档](https://github.com/volcengine/verl)

## 许可证

Apache License 2.0
