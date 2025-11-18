"""
Example reward functions for GroupAwareRewardManager - Proactive Agent Version.

These examples demonstrate how to implement adaptive reward scoring that considers
all responses generated for the same prompt when computing rewards, specifically
tailored for proactive agents that use <think> and <proactive> tags.
"""

import re
import numpy as np


def extract_formal_answer(response: str) -> str:
    """
    Extract the formal answer from a response by removing <think> and <proactive> tags.

    The response format is expected to be:
    - Optional <think>...</think> sections
    - Optional <proactive>...</proactive> sections
    - Formal answer (everything outside the tags)

    Args:
        response: The full response string

    Returns:
        The formal answer with think and proactive sections removed
    """
    # Remove <think>...</think> sections
    formal_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

    # Remove <proactive>...</proactive> sections
    formal_answer = re.sub(r'<proactive>.*?</proactive>', '', formal_answer, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace
    formal_answer = ' '.join(formal_answer.split())

    return formal_answer.strip()


def proactive_group_aware_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    beta: float = 0.8,
    **kwargs,
) -> list[float]:
    """
    Proactive group-aware reward function with improved correctness checking.

    Scoring rules:
    1. Correctness: +1 if ground truth in FORMAL ANSWER (excluding <think> and <proactive>), else 0
    2. Format bonus: +0.05 for <think>, +0.05 for <proactive>
    3. Proactive content: If has proactive content, score = beta * (1 - group_acc)
    4. Final: correctness + format_bonus OR proactive_score (whichever applies)

    Args:
        prompt_str: The prompt string (same for all responses in the group)
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts
        beta: Weight for proactive scoring (default: 0.8)

    Returns:
        List of reward scores, one per response
    """

    # Step 1: Compute correctness for each response
    # IMPORTANT: Only check correctness in the formal answer part
    correctness_scores = []
    for response, gt in zip(responses, ground_truths):
        # Extract formal answer (remove think and proactive tags)
        formal_answer = extract_formal_answer(response)

        # Check if ground truth is in the formal answer
        is_correct = gt.strip().lower() in formal_answer.strip().lower()
        correctness_scores.append(1.0 if is_correct else 0.0)

    # Step 2: Compute group accuracy based on formal answers
    group_acc = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0

    # Step 3: Compute rewards for each response
    rewards = []
    for response, correctness in zip(responses, correctness_scores):
        reward = 0.0

        # Check format bonuses (check in full response)
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
        has_proactive = bool(re.search(r'<proactive>.*?</proactive>', response, re.DOTALL | re.IGNORECASE))

        format_bonus = 0.0
        if has_think:
            format_bonus += 0.05
        if has_proactive:
            format_bonus += 0.05

        # If has proactive content, use proactive scoring
        if has_proactive:
            # Proactive responses get rewarded based on group difficulty
            # Harder problems (low group_acc) get higher rewards
            reward = beta * (1 - group_acc) + format_bonus
        else:
            # Regular scoring: correctness + format bonus
            reward = correctness + format_bonus

        rewards.append(reward)

    return rewards


def proactive_group_aware_reward_detailed(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    beta: float = 0.8,
    think_bonus: float = 0.05,
    proactive_bonus: float = 0.05,
    **kwargs,
) -> list[dict]:
    """
    Enhanced version that returns detailed information for debugging and analysis.

    Returns a list of dicts with detailed scoring information.
    """

    # Step 1: Compute correctness for each response
    correctness_scores = []
    formal_answers = []

    for response, gt in zip(responses, ground_truths):
        # Extract formal answer
        formal_answer = extract_formal_answer(response)
        formal_answers.append(formal_answer)

        # Check correctness in formal answer only
        is_correct = gt.strip().lower() in formal_answer.strip().lower()
        correctness_scores.append(1.0 if is_correct else 0.0)

    # Step 2: Compute group accuracy
    group_acc = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0

    # Step 3: Compute detailed rewards
    results = []
    for i, (response, correctness) in enumerate(zip(responses, correctness_scores)):
        # Check format
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
        has_proactive = bool(re.search(r'<proactive>.*?</proactive>', response, re.DOTALL | re.IGNORECASE))

        format_bonus = 0.0
        if has_think:
            format_bonus += think_bonus
        if has_proactive:
            format_bonus += proactive_bonus

        # Compute final reward
        if has_proactive:
            base_reward = beta * (1 - group_acc)
            reward = base_reward + format_bonus
        else:
            reward = correctness + format_bonus

        results.append({
            "score": reward,
            "correctness": correctness,
            "has_think": has_think,
            "has_proactive": has_proactive,
            "format_bonus": format_bonus,
            "group_acc": group_acc,
            "formal_answer": formal_answers[i][:100],  # First 100 chars for debugging
        })

    return results


def simple_group_aware_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    """
    Simple example: Compute basic correctness for each response, but the reward
    function has access to all responses in the group.

    This is a minimal example to show the interface. You could extend this to
    implement more sophisticated adaptive scoring strategies.

    Args:
        prompt_str: The prompt string (same for all responses in the group)
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers (usually the same for all)
        data_sources: List of data sources (usually the same for all)
        extra_infos: List of extra information dicts

    Returns:
        List of reward scores, one per response
    """
    scores = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        # Extract formal answer before checking
        formal_answer = extract_formal_answer(response)
        # Simple exact match (you would use more sophisticated logic here)
        if ground_truth.strip().lower() in formal_answer.strip().lower():
            score = 1.0
        else:
            score = 0.0
        scores.append(score)

    return scores


def adaptive_difficulty_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    """
    Adaptive reward based on group difficulty.

    If most responses in the group are correct, the problem is considered easier,
    and correct responses get lower rewards. If most are incorrect, correct responses
    get higher rewards (harder problem).

    This implements a simple form of adaptive difficulty scoring.

    Args:
        prompt_str: The prompt string
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts

    Returns:
        List of adaptive reward scores
    """
    # First, compute binary correctness for all responses
    is_correct = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)
        correct = ground_truth.strip().lower() in formal_answer.strip().lower()
        is_correct.append(correct)

    # Compute group success rate (difficulty measure)
    success_rate = sum(is_correct) / len(is_correct) if is_correct else 0.5

    # Adaptive reward: harder problems (lower success rate) get higher rewards
    # Difficulty bonus ranges from 0.5 to 1.5
    difficulty_multiplier = 2.0 - success_rate

    # Compute final scores
    scores = []
    for correct in is_correct:
        base_score = 1.0 if correct else 0.0
        adaptive_score = base_score * difficulty_multiplier
        scores.append(adaptive_score)

    return scores


def relative_quality_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[dict]:
    """
    Reward based on relative quality within the group.

    This example computes a quality score for each response (e.g., based on length,
    correctness, etc.), then assigns rewards based on the relative ranking within
    the group. This encourages diversity and competition among responses.

    Args:
        prompt_str: The prompt string
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts

    Returns:
        List of dicts with 'score' and additional metadata
    """
    # Compute quality scores (this is a simplified example)
    quality_scores = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)

        # Example quality metrics
        has_answer = ground_truth.strip().lower() in formal_answer.strip().lower()
        length_score = min(len(formal_answer) / 100.0, 1.0)  # Normalize length
        quality = (1.0 if has_answer else 0.0) + length_score * 0.2

        quality_scores.append(quality)

    # Compute mean and std of quality scores in this group
    mean_quality = np.mean(quality_scores)
    std_quality = np.std(quality_scores) if len(quality_scores) > 1 else 1.0

    # Assign rewards based on relative performance
    results = []
    for i, quality in enumerate(quality_scores):
        # Normalize by group statistics
        if std_quality > 0:
            normalized_score = (quality - mean_quality) / std_quality
        else:
            normalized_score = 0.0

        # Convert to reward (you can adjust the scaling)
        reward = normalized_score

        results.append(
            {
                "score": reward,
                "quality_score": quality,
                "group_mean_quality": mean_quality,
                "group_std_quality": std_quality,
            }
        )

    return results


def best_of_n_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    """
    Best-of-N style reward: Only the best response(s) in the group get positive reward.

    This implements a winner-takes-all or winner-takes-most strategy where only
    the top-performing responses get rewards, encouraging the model to produce
    at least one high-quality response per prompt.

    Args:
        prompt_str: The prompt string
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts

    Returns:
        List of reward scores (only top responses get positive rewards)
    """
    # Compute quality for each response
    qualities = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)

        # Example quality metric (replace with your own)
        has_answer = ground_truth.strip().lower() in formal_answer.strip().lower()
        quality = 1.0 if has_answer else 0.0

        qualities.append(quality)

    # Find the maximum quality in the group
    max_quality = max(qualities) if qualities else 0.0

    # Only reward responses that match the best quality
    # (you could also reward top-k instead)
    scores = []
    for quality in qualities:
        if quality == max_quality and max_quality > 0:
            # Winner gets reward
            score = 1.0
        else:
            # Losers get no reward (or small penalty)
            score = 0.0

        scores.append(score)

    return scores


def pass_at_k_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    k: int = 1,
    **kwargs,
) -> list[dict]:
    """
    Pass@k style reward for code generation or similar tasks.

    Rewards all responses in the group based on whether at least k responses
    in the group are correct. This encourages the model to generate multiple
    correct solutions.

    Args:
        prompt_str: The prompt string
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts
        k: Minimum number of correct responses needed for the group to succeed

    Returns:
        List of dicts with rewards and metadata
    """
    # Compute correctness for each response
    is_correct = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)
        # Replace with actual test execution for code
        correct = ground_truth.strip().lower() in formal_answer.strip().lower()
        is_correct.append(correct)

    # Count number of correct responses
    num_correct = sum(is_correct)
    group_passes = num_correct >= k

    # Assign rewards
    results = []
    for i, correct in enumerate(is_correct):
        if group_passes:
            # Group passed: reward correct responses more
            reward = 1.0 if correct else -0.1
        else:
            # Group failed: smaller rewards
            reward = 0.5 if correct else -0.2

        results.append(
            {
                "score": reward,
                "is_correct": correct,
                "num_correct_in_group": num_correct,
                "group_passes": group_passes,
            }
        )

    return results


# You can add more sophisticated reward functions here, such as:
# - Diversity-aware rewards (penalize similar responses)
# - Curriculum learning rewards (adapt based on training progress)
# - Multi-objective rewards (combining multiple metrics)
# - Uncertainty-based rewards (using model confidence)
