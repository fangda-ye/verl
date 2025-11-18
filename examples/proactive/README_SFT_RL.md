# SFT Warmup + RL Training Pipeline

完整的训练流程：先用 SFT 数据做冷启动，然后继续 RL 训练。

## 📊 数据说明

### 两种数据格式

#### 1. SFT 数据（`sampleQA_processed_2.jsonl`）

包含完整的对话，带有 `<think>` 和 `<proactive>` 标签：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful proactive assistant."},
    {"role": "user", "content": "How much money, in euros, was the surgeon...?"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n<proactive>\n...\n</proactive>\n\nI don't have the exact financial details..."}
  ],
  "id": 5,
  "sub_category": "simpleQA"
}
```

**用途**：
- 教会模型使用 `<think>` 和 `<proactive>` 标签
- 冷启动训练，提供初始能力
- 随机抽取 50 条用于 SFT

#### 2. RL 数据（`sampleQA.jsonl`）

只包含 user 问题和简单答案（ground truth）：

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

**用途**：
- RL 训练的 prompt 来源
- 只使用 user 问题作为 prompt
- assistant 答案作为 ground truth 用于 reward 计算

## 🚀 使用方法

### 方式 1：一键运行（推荐）

完整流程自动化：SFT → RL

```bash
bash examples/proactive/run_sft_then_rl.sh
```

脚本会自动：
1. ✅ 从 `data/sampleQA_processed_2.jsonl` 随机抽取 50 条
2. ✅ 运行 SFT 训练（2 epochs）
3. ✅ 加载 SFT checkpoint
4. ✅ 处理 `data/sampleQA.jsonl` 用于 RL
5. ✅ 运行 RL 训练（10 epochs）

### 方式 2：分步运行

#### 步骤 1：SFT 冷启动

```bash
bash examples/proactive/run_sft_warmup.sh
```

这会：
- 随机抽取 50 条 SFT 数据
- 训练 2 epochs
- 保存 checkpoint 到 `checkpoints/proactive_sft_warmup/`

#### 步骤 2：RL 训练

使用 SFT checkpoint 继续训练：

```bash
# 方式 A：手动指定 checkpoint
bash examples/proactive/run_proactive_grpo.sh \
    actor_rollout_ref.model.path='checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2'

# 方式 B：从基础模型开始（不推荐，跳过 SFT）
bash examples/proactive/run_proactive_grpo.sh
```

## 📁 文件结构

```
examples/proactive/
├── process_sft_data.py              # 处理 SFT 数据（随机抽样）
├── process_sampleQA.py              # 处理 RL 数据
├── group_aware_reward.py            # 奖励函数
├── run_sft_warmup.sh                # 只运行 SFT
├── run_proactive_grpo.sh            # 只运行 RL
├── run_sft_then_rl.sh               # 完整流程（推荐）
└── README_SFT_RL.md                 # 本文档

data/
├── sampleQA_processed_2.jsonl       # SFT 数据源（带标签的完整回答）
├── sampleQA.jsonl                   # RL 数据源（简单 QA）
├── sft_samples.jsonl                # 抽取的 50 条 SFT 数据
└── processed_sampleQA/              # 处理后的 RL 数据
    ├── train.parquet
    └── test.parquet

checkpoints/
├── proactive_sft_warmup/            # SFT checkpoints
│   └── sft_50_samples/
│       └── checkpoints/
│           └── epoch_2/             # 用这个继续 RL
└── proactive_agent/                 # RL checkpoints
    └── sft50_rl_beta0.5_n16/
```

## ⚙️ 配置参数

### SFT 配置

在 `run_sft_then_rl.sh` 中修改：

```bash
NUM_SFT_SAMPLES=50      # SFT 数据数量（默认 50）
SFT_EPOCHS=2            # SFT 训练轮数（默认 2）
```

### RL 配置

```bash
RL_BETA=0.5             # Proactive 奖励权重（0.0-1.0）
RL_ROLLOUTS=16          # 每个 prompt 的 rollouts 数量
RL_EPOCHS=10            # RL 训练轮数
```

### 模型路径

```bash
BASE_MODEL_PATH="/mnt/hdd/Fangda/data/models/qwen3-8b"
```

## 🎯 训练流程详解

### Stage 1: SFT Warmup

**目标**：教会模型使用 `<think>` 和 `<proactive>` 标签

**数据**：
- 输入：50 条完整的对话（包含标签）
- 训练方式：标准的 supervised fine-tuning

**输出**：
- Checkpoint：`checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2/`
- 模型已经学会：
  - 使用 `<think>` 标签思考
  - 使用 `<proactive>` 标签做主动推理
  - 基本的回答格式

### Stage 2: RL Training

**目标**：通过 group-aware rewards 优化生成质量

**数据**：
- 输入：只有 user 问题（无标签示例）
- Ground truth：简单答案（用于 reward 计算）

**奖励策略**：
- 正确性：只检查正式回答部分（移除标签内容）
- 格式奖励：使用 `<think>` 和 `<proactive>` 标签
- Proactive 奖励：难题（低 group acc）获得更高奖励

**输出**：
- Checkpoint：`checkpoints/proactive_agent/sft50_rl_beta0.5_n16/`
- 模型已经学会：
  - 在合适的时候使用 proactive 思考
  - 根据问题难度调整策略
  - 生成高质量的结构化回答

## 📊 为什么需要 SFT 冷启动？

### 问题：直接 RL 训练

如果直接从基础模型开始 RL：
- ❌ 模型不知道 `<think>` 和 `<proactive>` 标签
- ❌ 无法生成结构化的回答
- ❌ RL 训练效果差，难以收敛

### 解决方案：SFT → RL

先用 SFT 教会模型格式：
- ✅ 模型学会使用标签
- ✅ 理解什么是 proactive 思考
- ✅ RL 训练可以在良好的基础上优化

## 🔧 自定义参数

### 修改 SFT 样本数量

```bash
bash examples/proactive/run_sft_then_rl.sh
# 编辑脚本，修改 NUM_SFT_SAMPLES=100
```

### 修改 RL 参数

```bash
# 修改 beta 值
bash examples/proactive/run_sft_then_rl.sh \
    custom_reward_function.reward_kwargs.beta=0.7

# 修改 rollout 数量
bash examples/proactive/run_sft_then_rl.sh \
    actor_rollout_ref.rollout.n=8

# 修改训练轮数
bash examples/proactive/run_sft_then_rl.sh \
    trainer.total_epochs=20
```

### 只运行 SFT（不接 RL）

```bash
bash examples/proactive/run_sft_warmup.sh
```

### 从 SFT checkpoint 开始 RL

```bash
SFT_CKPT="checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2"

bash examples/proactive/run_proactive_grpo.sh \
    actor_rollout_ref.model.path="${SFT_CKPT}"
```

## 📈 监控训练

### Tensorboard

```bash
# SFT 训练
tensorboard --logdir checkpoints/proactive_sft_warmup/

# RL 训练
tensorboard --logdir checkpoints/proactive_agent/
```

### 检查 Checkpoints

```bash
# SFT checkpoints
ls -lh checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/

# RL checkpoints
ls -lh checkpoints/proactive_agent/sft50_rl_beta0.5_n16/
```

## 🎓 实验建议

### 对比实验

#### 实验 1：不同 SFT 样本数量

```bash
# 25 samples
NUM_SFT_SAMPLES=25 bash examples/proactive/run_sft_then_rl.sh

# 50 samples（推荐）
NUM_SFT_SAMPLES=50 bash examples/proactive/run_sft_then_rl.sh

# 100 samples
NUM_SFT_SAMPLES=100 bash examples/proactive/run_sft_then_rl.sh
```

#### 实验 2：有无 SFT 冷启动

```bash
# 有 SFT（推荐）
bash examples/proactive/run_sft_then_rl.sh

# 无 SFT（baseline）
bash examples/proactive/run_proactive_grpo.sh
```

#### 实验 3：不同 beta 值

```bash
# 低 beta（不鼓励 proactive）
bash examples/proactive/run_sft_then_rl.sh \
    custom_reward_function.reward_kwargs.beta=0.3

# 中 beta（平衡）
bash examples/proactive/run_sft_then_rl.sh \
    custom_reward_function.reward_kwargs.beta=0.5

# 高 beta（强烈鼓励 proactive）
bash examples/proactive/run_sft_then_rl.sh \
    custom_reward_function.reward_kwargs.beta=0.7
```

## ⚠️ 常见问题

### Q: SFT 数据不够怎么办？

A: 调整 `NUM_SFT_SAMPLES`，或者准备更多带标签的数据。最少建议 25 条。

### Q: SFT 训练很快就结束了？

A: 正常。50 条数据，2 epochs，4 GPU 训练会很快（几分钟）。这只是冷启动。

### Q: 如何知道 SFT 是否成功？

A: 检查 SFT checkpoint 的生成：
```bash
# 手动测试生成
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/proactive_sft_warmup/sft_50_samples/checkpoints/epoch_2')
# 测试生成...
"
```

### Q: RL 训练可以跳过 SFT 吗？

A: 可以，但不推荐。没有 SFT，模型不知道如何使用标签，RL 效果会差很多。

### Q: 两个数据集可以合并吗？

A: 不建议。它们用途不同：
- SFT 数据：教格式和基础能力
- RL 数据：优化生成策略

## 🔍 验证结果

训练完成后，生成应该类似：

```
User: How much money was...?

Model:
<think>
This is a specific legal case question...
</think>

<proactive>
I don't have access to detailed legal records...
</proactive>

I don't have the exact financial details. Please consult legal records.
```

关键指标：
- ✅ 使用了 `<think>` 和 `<proactive>` 标签
- ✅ 结构清晰
- ✅ Proactive 内容合理（承认不确定性）
- ✅ 正式回答简洁明了

## 📚 参考

- [Group-Aware Reward Manager](./README.md)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [veRL Documentation](https://github.com/volcengine/verl)
