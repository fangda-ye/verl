# Group-Aware Reward Scoring for GRPO

本目录包含使用 `GroupAwareRewardManager` 实现自适应奖励评分（adaptive reward scoring）的示例和配置。

## 什么是 Group-Aware Reward Scoring？

在 GRPO 训练中，每个 prompt 会生成多个 responses（通过 `rollout.n` 参数配置）。传统的奖励计算方式是独立评估每个 response，但有时我们希望：

- **看到同一个 prompt 的所有 responses** 的表现
- **基于 group 的统计信息** 来调整每个 response 的奖励
- **实现自适应的奖励策略**，例如：
  - 难题（大多数回答都错）给予更高奖励
  - 基于相对质量排名给予奖励
  - 只奖励 best-of-n
  - Pass@k 风格的奖励

`GroupAwareRewardManager` 就是为了解决这个需求而设计的。

## 核心区别

### 传统的 NaiveRewardManager

```python
# 每次只看一个 response
for response in responses:
    reward = compute_score(response, ground_truth)
```

### GroupAwareRewardManager

```python
# 一次性看到同一个 prompt 的所有 responses
all_responses = get_all_responses_for_prompt(prompt_uid)
rewards = compute_score(
    prompt_str=prompt,
    responses=all_responses,  # 所有 responses
    ground_truths=[gt] * len(all_responses),
    ...
)
```

## 文件说明

- **`group_aware_reward_examples.py`** - 5 个示例奖励函数，展示不同的自适应奖励策略
- **`config_with_group_aware_reward.yaml`** - 完整的训练配置示例
- **`README_GROUP_AWARE_REWARD.md`** - 本文档

## 快速开始

### 1. 选择或编写奖励函数

从 `group_aware_reward_examples.py` 选择一个，或编写自己的：

```python
def my_adaptive_reward(
    prompt_str: str,
    responses: list[str],      # 同一个 prompt 的所有 responses
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs
) -> list[float]:
    """返回每个 response 的奖励分数"""

    # 你的逻辑：可以基于所有 responses 的表现来计算
    scores = []
    for response in responses:
        score = compute_your_score(response, all_responses=responses)
        scores.append(score)

    return scores
```

### 2. 配置训练

修改你的配置文件：

```yaml
# 使用 group_aware reward manager
reward_model:
  reward_manager: group_aware

# 指定自定义奖励函数
custom_reward_function:
  path: /path/to/your_reward_function.py
  name: my_adaptive_reward

# GRPO 设置
algorithm:
  adv_estimator: grpo

# 每个 prompt 生成多个 responses
actor_rollout_ref:
  rollout:
    n: 8  # 例如：8 个 responses per prompt
```

### 3. 运行训练

```bash
python -m verl.trainer.main_ppo \
    --config-path examples/data_parallel/grpo \
    --config-name config_with_group_aware_reward \
    ++custom_reward_function.path=/path/to/your_reward.py \
    ++custom_reward_function.name=your_function_name
```

## 示例奖励函数说明

### 1. `simple_group_aware_reward`
最简单的示例，展示接口用法。基础的正确性检查。

**适用场景**：学习接口、快速测试

### 2. `adaptive_difficulty_reward`
根据 group 的成功率调整奖励：
- 成功率低（难题）→ 更高奖励
- 成功率高（简单题）→ 较低奖励

**适用场景**：平衡训练难度、curriculum learning

### 3. `relative_quality_reward`
基于 group 内的相对质量排名给予奖励（归一化）。

**适用场景**：鼓励多样性、相对性能优化

### 4. `best_of_n_reward`
只奖励 group 中最好的 response(s)。

**适用场景**：winner-takes-all、注重质量而非数量

### 5. `pass_at_k_reward`
基于 group 中至少有 k 个正确答案来分配奖励。

**适用场景**：代码生成、需要多个正确解的任务

## 工作原理

```
1. Rollout 阶段：
   Prompt A → [Response A1, Response A2, ..., Response A8]
   Prompt B → [Response B1, Response B2, ..., Response B8]

2. Grouping（自动完成）：
   根据 UID 将 responses 分组

3. Group-Aware Reward 计算：
   Group A: compute_score(prompt_A, [A1, A2, ..., A8], ...)
            → returns [reward_A1, reward_A2, ..., reward_A8]

   Group B: compute_score(prompt_B, [B1, B2, ..., B8], ...)
            → returns [reward_B1, reward_B2, ..., reward_B8]

4. GRPO Advantage 计算：
   使用 rewards 和 grouping 计算 advantages
```

## 调试技巧

### 打印 Group 信息

在配置中设置 `num_examine`:

```yaml
reward_model:
  reward_manager: group_aware
  num_examine: 5  # 打印前 5 个 groups
```

输出示例：
```
[Group UID: abc123]
[prompt] What is 2+2?
[num_responses] 8

  [response_0] The answer is 4.
  [score_0] 1.2

  [response_1] I think it's 5.
  [score_1] 0.0

  ...
```

### 返回额外信息

奖励函数可以返回字典来追踪更多信息：

```python
return [
    {
        "score": 1.0,  # 必需
        "quality": 0.8,
        "is_correct": True,
        "group_mean": 0.6,
    }
    for response in responses
]
```

## 常见问题

### Q: 和标准 GRPO 有什么区别？

A: 标准 GRPO 也会按 group 计算 advantage，但奖励计算是独立的。GroupAwareRewardManager 让奖励计算也能看到整个 group。

### Q: 需要修改 GRPO 的其他配置吗？

A: 不需要。只需要：
1. 设置 `reward_model.reward_manager: group_aware`
2. 提供自定义奖励函数
3. 确保 `rollout.n > 1`

### Q: 性能影响？

A: 几乎没有。我们只是改变了奖励函数的调用方式，从 N 次单独调用变成按 group 批量调用。

### Q: 可以用于非 GRPO 算法吗？

A: 可以，但主要价值在于 GRPO。因为 GRPO 本身就需要 `rollout.n > 1`。

## 进阶用法

### 自定义参数

通过配置传递额外参数：

```yaml
custom_reward_function:
  path: my_reward.py
  name: my_function
  reward_kwargs:
    alpha: 0.5
    beta: 1.0
    custom_threshold: 0.8
```

在奖励函数中接收：

```python
def my_function(
    prompt_str, responses, ground_truths, data_sources, extra_infos,
    alpha=0.5, beta=1.0, custom_threshold=0.8,  # 从 config 传入
    **kwargs
):
    # 使用这些参数
    ...
```

### 组合多个奖励

```python
def combined_reward(prompt_str, responses, ...):
    # 基础正确性
    correctness_scores = [check_correctness(r) for r in responses]

    # 相对质量
    quality_scores = compute_relative_quality(responses)

    # 组合
    final_scores = [
        0.7 * c + 0.3 * q
        for c, q in zip(correctness_scores, quality_scores)
    ]

    return final_scores
```

## 参考文档

- 完整文档：`/home/user/verl/docs/workers/group_aware_reward_manager.md`
- GRPO 论文：https://arxiv.org/abs/2402.03300
- veRL 文档：https://github.com/volcengine/verl

## 联系和反馈

如有问题或建议，欢迎提 issue 或 PR。
