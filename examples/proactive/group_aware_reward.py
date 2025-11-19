"""
Example reward functions for GroupAwareRewardManager - Proactive Agent Version.

These examples demonstrate how to implement adaptive reward scoring that considers
all responses generated for the same prompt when computing rewards, specifically
tailored for proactive agents that use <think> and <proactive> tags.
"""

import re
from difflib import SequenceMatcher
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


def extract_proactive_content(response: str) -> str:
    """
    Extract the content inside <proactive>...</proactive> tags.

    Args:
        response: The full response string

    Returns:
        The proactive content (empty string if no proactive tag or empty content)
    """
    match = re.search(r'<proactive>(.*?)</proactive>', response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        return content
    return ""


def normalize_text(text: str) -> str:
    """
    Normalize text for better matching.
    - Convert to lowercase
    - Remove extra whitespace
    - Remove common punctuation
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def fuzzy_match(text: str, target: str, threshold: float = 0.6) -> bool:
    """
    Check if target appears in text with fuzzy matching.

    Args:
        text: The text to search in
        target: The target string to find
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        True if target is found in text with similarity >= threshold
    """
    text_norm = normalize_text(text)
    target_norm = normalize_text(target)

    # Exact match after normalization
    if target_norm in text_norm:
        return True

    # Fuzzy match - check if any substring has high similarity
    target_len = len(target_norm)
    if target_len == 0:
        return False

    # Sliding window fuzzy match
    for i in range(len(text_norm) - target_len + 1):
        substring = text_norm[i:i + target_len]
        similarity = SequenceMatcher(None, substring, target_norm).ratio()
        if similarity >= threshold:
            return True

    # Also check overall similarity for short targets
    if len(target_norm.split()) <= 3:
        similarity = SequenceMatcher(None, text_norm, target_norm).ratio()
        if similarity >= threshold:
            return True

    return False


def check_correctness(formal_answer: str, ground_truth: str) -> bool:
    """
    Check if the formal answer is correct using multiple matching strategies.

    Args:
        formal_answer: The extracted formal answer
        ground_truth: The ground truth answer

    Returns:
        True if the answer is considered correct
    """
    # Strategy 1: Exact substring match (normalized)
    if normalize_text(ground_truth) in normalize_text(formal_answer):
        return True

    # Strategy 2: Fuzzy match with 80% threshold
    if fuzzy_match(formal_answer, ground_truth, threshold=0.8):
        return True

    # Strategy 3: For numeric answers, extract and compare numbers
    gt_numbers = re.findall(r'\d+\.?\d*', ground_truth)
    answer_numbers = re.findall(r'\d+\.?\d*', formal_answer)
    if gt_numbers and answer_numbers:
        # Check if all GT numbers appear in the answer
        if all(num in answer_numbers for num in gt_numbers):
            return True

    return False


def proactive_group_aware_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    beta: float = 0.8,
    min_proactive_length: int = 10,
    **kwargs,
) -> list[float]:
    """
    Proactive group-aware reward function with improved correctness checking.

    Scoring rules:
    1. If <proactive> has substantial content (>= min_proactive_length chars):
       - This is a "don't know" answer
       - Reward = beta * (1 - group_acc) + format_bonus
       - Do NOT check correctness (we assume proactive means uncertain)

    2. If NO substantial <proactive> content:
       - Check correctness using improved matching
       - Reward = correctness + format_bonus

    Format bonuses:
    - +0.05 for <think> tag
    - +0.05 for <proactive> tag (only if it has content)

    Args:
        prompt_str: The prompt string (same for all responses in the group)
        responses: List of all response strings for this prompt
        ground_truths: List of ground truth answers
        data_sources: List of data sources
        extra_infos: List of extra information dicts
        beta: Weight for proactive scoring (default: 0.8)
        min_proactive_length: Minimum chars in proactive to count as substantial (default: 10)

    Returns:
        List of reward scores, one per response
    """

    # Step 1: First pass - compute correctness for responses WITHOUT substantial proactive content
    correctness_scores = []
    has_substantial_proactive = []

    for response, gt in zip(responses, ground_truths):
        # Extract proactive content
        proactive_content = extract_proactive_content(response)
        has_proactive = len(proactive_content) >= min_proactive_length

        has_substantial_proactive.append(has_proactive)

        if has_proactive:
            # Proactive response - don't compute correctness
            correctness_scores.append(0.0)  # Placeholder, won't be used
        else:
            # Regular response - check correctness
            formal_answer = extract_formal_answer(response)
            is_correct = check_correctness(formal_answer, gt)
            correctness_scores.append(1.0 if is_correct else 0.0)

    # Step 2: Compute group accuracy (only from non-proactive responses)
    non_proactive_correctness = [
        score for score, has_pro in zip(correctness_scores, has_substantial_proactive)
        if not has_pro
    ]
    if non_proactive_correctness:
        group_acc = sum(non_proactive_correctness) / len(non_proactive_correctness)
    else:
        # All responses are proactive - set group_acc to 0 (hardest)
        group_acc = 0.0

    # Step 3: Compute rewards for each response
    rewards = []
    for i, (response, correctness, has_proactive) in enumerate(
        zip(responses, correctness_scores, has_substantial_proactive)
    ):
        # Check format tags
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
        proactive_content = extract_proactive_content(response)

        # Format bonuses
        format_bonus = 0.0
        if has_think:
            format_bonus += 0.05
        if len(proactive_content) >= min_proactive_length:
            format_bonus += 0.05

        # Compute reward based on response type
        if has_proactive:
            # Proactive response: reward based on group difficulty
            # Harder problems (low group_acc) get higher rewards
            reward = beta * (1 - group_acc) + format_bonus
        else:
            # Regular response: reward based on correctness
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
    min_proactive_length: int = 10,
    think_bonus: float = 0.05,
    proactive_bonus: float = 0.05,
    **kwargs,
) -> list[dict]:
    """
    Enhanced version that returns detailed information for debugging and analysis.

    Returns a list of dicts with detailed scoring information.
    """

    # Step 1: Compute correctness
    correctness_scores = []
    has_substantial_proactive = []
    formal_answers = []
    proactive_contents = []

    for response, gt in zip(responses, ground_truths):
        # Extract proactive content
        proactive_content = extract_proactive_content(response)
        proactive_contents.append(proactive_content)
        has_proactive = len(proactive_content) >= min_proactive_length
        has_substantial_proactive.append(has_proactive)

        # Extract formal answer
        formal_answer = extract_formal_answer(response)
        formal_answers.append(formal_answer)

        if has_proactive:
            correctness_scores.append(0.0)
        else:
            is_correct = check_correctness(formal_answer, gt)
            correctness_scores.append(1.0 if is_correct else 0.0)

    # Step 2: Compute group accuracy
    non_proactive_correctness = [
        score for score, has_pro in zip(correctness_scores, has_substantial_proactive)
        if not has_pro
    ]
    if non_proactive_correctness:
        group_acc = sum(non_proactive_correctness) / len(non_proactive_correctness)
    else:
        group_acc = 0.0

    # Step 3: Compute detailed rewards
    results = []
    for i, (response, correctness, has_proactive) in enumerate(
        zip(responses, correctness_scores, has_substantial_proactive)
    ):
        # Check format
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
        proactive_content = proactive_contents[i]

        format_bonus = 0.0
        if has_think:
            format_bonus += think_bonus
        if len(proactive_content) >= min_proactive_length:
            format_bonus += proactive_bonus

        # Compute final reward
        if has_proactive:
            base_reward = beta * (1 - group_acc)
            reward = base_reward + format_bonus
            reward_type = "proactive"
        else:
            reward = correctness + format_bonus
            reward_type = "correctness"

        results.append({
            "score": reward,
            "reward_type": reward_type,
            "correctness": correctness,
            "has_think": has_think,
            "has_substantial_proactive": has_proactive,
            "proactive_content_length": len(proactive_content),
            "format_bonus": format_bonus,
            "group_acc": group_acc,
            "formal_answer": formal_answers[i][:100],  # First 100 chars for debugging
            "proactive_preview": proactive_content[:100] if proactive_content else "",
        })

    return results


# Keep other reward functions for compatibility
def simple_group_aware_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    """Simple example with improved matching."""
    scores = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)
        is_correct = check_correctness(formal_answer, ground_truth)
        scores.append(1.0 if is_correct else 0.0)
    return scores


def adaptive_difficulty_reward(
    prompt_str: str,
    responses: list[str],
    ground_truths: list[str],
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs,
) -> list[float]:
    """Adaptive reward with improved matching."""
    is_correct = []
    for i, response in enumerate(responses):
        ground_truth = ground_truths[i]
        formal_answer = extract_formal_answer(response)
        correct = check_correctness(formal_answer, ground_truth)
        is_correct.append(correct)

    success_rate = sum(is_correct) / len(is_correct) if is_correct else 0.5
    difficulty_multiplier = 2.0 - success_rate

    scores = []
    for correct in is_correct:
        base_score = 1.0 if correct else 0.0
        adaptive_score = base_score * difficulty_multiplier
        scores.append(adaptive_score)

    return scores
