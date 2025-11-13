# GroupAwareRewardManager - Adaptive Reward Scoring for GRPO

## Overview

`GroupAwareRewardManager` is a reward manager designed for scenarios where multiple responses are generated per prompt (e.g., GRPO training), and you want to compute rewards with awareness of all responses in the same group. This enables **adaptive reward scoring** based on the relative performance of all rollouts.

## Why Use GroupAwareRewardManager?

In standard reward computation, each response is evaluated independently. However, in GRPO:
- Multiple responses (e.g., 4-16) are generated per prompt
- GRPO computes advantages by comparing responses within each group
- But reward computation happens **before** the grouping, so each response is scored independently

`GroupAwareRewardManager` changes this by:
1. **Grouping responses by their prompt UID** before reward computation
2. **Passing all responses in a group** to your custom reward function
3. Allowing **adaptive scoring** based on group-level statistics

## Use Cases

### 1. Adaptive Difficulty Scoring
Adjust rewards based on how many responses in the group are correct:
- If most responses are correct → easier problem → lower rewards
- If most responses are incorrect → harder problem → higher rewards

### 2. Relative Quality Scoring
Reward responses based on their relative quality within the group:
- Encourages diversity and competition among responses
- Normalizes rewards by group statistics

### 3. Best-of-N Rewards
Only reward the best response(s) in each group:
- Winner-takes-all or winner-takes-most strategy
- Encourages at least one high-quality response per prompt

### 4. Pass@k Style Rewards
Reward based on whether at least k responses in the group are correct:
- Common in code generation tasks
- Encourages multiple correct solutions

### 5. Diversity-Aware Rewards
Penalize similar or redundant responses within a group:
- Encourages exploration of different solution strategies
- Reduces mode collapse

## Configuration

### Step 1: Create a Custom Reward Function

Create a Python file with your group-aware reward function:

```python
# my_reward_function.py

def adaptive_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs
) -> list[float]:
    """
    Compute adaptive rewards based on group statistics.

    Args:
        prompt_str: The prompt string (same for all responses)
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts

    Returns:
        List of reward scores, one per response
    """
    # Compute base scores
    scores = []
    for i, response in enumerate(responses):
        # Your scoring logic here
        score = compute_score(response, ground_truths[i])
        scores.append(score)

    # Adaptive adjustment based on group performance
    mean_score = sum(scores) / len(scores)

    # Example: Boost rewards for harder problems (low success rate)
    difficulty_multiplier = 2.0 - mean_score  # Range: 1.0 to 2.0

    adaptive_scores = [s * difficulty_multiplier for s in scores]

    return adaptive_scores
```

You can also return a list of dictionaries with additional metadata:

```python
def advanced_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs
) -> list[dict]:
    """Returns detailed reward information."""

    results = []
    for i, response in enumerate(responses):
        score = compute_score(response, ground_truths[i])

        results.append({
            "score": score,  # Required field
            "quality": quality_metric(response),
            "correctness": is_correct(response, ground_truths[i]),
            # Add any other metrics you want to track
        })

    return results
```

### Step 2: Update Your Training Configuration

Modify your GRPO training config to use `GroupAwareRewardManager`:

```yaml
# config.yaml

# Use group_aware reward manager
reward_model:
  reward_manager: group_aware  # Changed from "naive" to "group_aware"

# Specify your custom reward function
custom_reward_function:
  path: /path/to/my_reward_function.py  # Path to your reward function file
  name: adaptive_reward  # Function name
  reward_kwargs:  # Optional kwargs passed to your function
    custom_param: value

# GRPO-specific settings
algorithm:
  adv_estimator: grpo  # Use GRPO advantage estimation

# Multiple rollouts per prompt (required for GRPO)
actor_rollout_ref:
  rollout:
    n: 8  # Number of responses per prompt (e.g., 4, 8, or 16)
```

### Step 3: Run Training

```bash
python -m verl.trainer.main_ppo \
    data=<your_data_config> \
    actor_rollout_ref=<your_model_config> \
    reward_model=<your_reward_config> \
    algorithm=grpo \
    ++reward_model.reward_manager=group_aware \
    ++custom_reward_function.path=/path/to/my_reward_function.py \
    ++custom_reward_function.name=adaptive_reward \
    ++actor_rollout_ref.rollout.n=8
```

## Example Reward Functions

See `examples/data_parallel/grpo/group_aware_reward_examples.py` for complete examples:

1. **`simple_group_aware_reward`** - Basic example showing the interface
2. **`adaptive_difficulty_reward`** - Adjust rewards based on group success rate
3. **`relative_quality_reward`** - Normalize rewards by group statistics
4. **`best_of_n_reward`** - Winner-takes-all strategy
5. **`pass_at_k_reward`** - Pass@k style rewards for code generation

## API Reference

### Custom Reward Function Signature

```python
def my_reward_function(
    prompt_str: str,           # The prompt (same for all responses)
    responses: list[str],      # All responses for this prompt
    ground_truths: list[str],  # Ground truths (usually same for all)
    data_sources: list[str],   # Data sources (usually same for all)
    extra_infos: list[dict],   # Extra metadata for each response
    **kwargs                   # Additional kwargs from config
) -> list[float] | list[dict]:
    """
    Returns:
        - list[float]: Simple scores, one per response
        - list[dict]: Dicts with "score" key + optional metadata
    """
    pass
```

### GroupAwareRewardManager Parameters

- **`tokenizer`**: Tokenizer for decoding token IDs
- **`num_examine`**: Number of groups to print for debugging (default: 0)
- **`compute_score`**: Your custom reward function (required)
- **`reward_fn_key`**: Key to access data source in batch (default: "data_source")

## How It Works

1. **Grouping**: Responses are grouped by their `uid` field (unique prompt identifier)
2. **Batch Processing**: For each group:
   - Collect all responses, ground truths, and metadata
   - Call your reward function with the complete group
3. **Reward Assignment**: Assign returned scores to the last token of each response
4. **GRPO Advantage Computation**: GRPO uses these rewards + grouping to compute advantages

## Differences from NaiveRewardManager

| Feature | NaiveRewardManager | GroupAwareRewardManager |
|---------|-------------------|------------------------|
| Processing | One response at a time | All responses in a group |
| Reward function input | Single response | List of responses |
| Use case | Independent scoring | Adaptive/relative scoring |
| GRPO support | ✓ (but no group awareness) | ✓ (group-aware scoring) |

## Best Practices

1. **Ensure consistent group sizes**: Set `actor_rollout_ref.rollout.n` appropriately
2. **Handle edge cases**: Check for groups with only 1 response
3. **Return the correct number of scores**: Must match the number of responses in the group
4. **Use metadata for debugging**: Return dicts with extra info to track group statistics
5. **Test your reward function**: Print debug output using `num_examine` parameter

## Debugging

Enable debug output by setting `num_examine` in your reward manager:

```python
# This will print the first 5 groups during training
reward_fn = GroupAwareRewardManager(
    tokenizer=tokenizer,
    num_examine=5,  # Print first 5 groups
    compute_score=my_reward_function
)
```

Output format:
```
[Group UID: abc123]
[prompt] What is 2+2?
[num_responses] 4

  [response_0] The answer is 4.
  [ground_truth_0] 4
  [score_0] 1.0

  [response_1] I think it's 5.
  [ground_truth_1] 4
  [score_1] 0.0
  ...
```

## Limitations

- Requires unique `uid` field in the data (automatically added by the trainer)
- Custom reward function must be provided (no default implementation)
- Slightly higher memory usage due to grouping (minimal impact)

## Further Reading

- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [veRL Documentation](../README.md)
