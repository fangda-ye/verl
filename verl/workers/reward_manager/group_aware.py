# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("group_aware")
class GroupAwareRewardManager(AbstractRewardManager):
    """
    A reward manager that groups responses by their prompt UID and computes rewards
    with access to all responses in the same group. This enables adaptive reward scoring
    based on the performance of all rollouts for the same prompt.

    This is particularly useful for GRPO where multiple responses are generated per prompt,
    and you want the reward function to consider the relative performance of all responses
    when assigning rewards.

    The custom reward function should have the signature:
        def compute_score(
            prompt_str: str,
            responses: list[str],  # All responses for this prompt
            ground_truths: list[str],  # Ground truths for all responses (usually the same)
            data_sources: list[str],  # Data sources for all responses (usually the same)
            extra_infos: list[dict],  # Extra info for all responses
            **kwargs
        ) -> list[float] | list[dict]:
            # Returns a score for each response, considering all responses in the group
            # Can return either a list of floats or a list of dicts with 'score' key
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the GroupAwareRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of groups of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score that takes all responses in a group.
                          Must accept prompt_str, responses (list), ground_truths (list), data_sources (list),
                          and extra_infos (list) as arguments.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
                          Defaults to "data_source".
        """
        if compute_score is None:
            raise ValueError(
                "GroupAwareRewardManager requires a custom compute_score function. "
                "Please provide one via config.custom_reward_function"
            )

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for all responses, grouped by their prompt UID.

        This method groups responses by their 'uid' field, then calls the compute_score
        function with all responses in each group. This allows the reward function to
        implement adaptive scoring based on the relative performance of all responses.
        """

        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Group responses by UID
        uid_to_indices = defaultdict(list)
        for i in range(len(data)):
            uid = data[i].non_tensor_batch.get("uid", str(i))
            uid_to_indices[uid].append(i)

        examined_groups = 0

        # Process each group
        for uid, indices in uid_to_indices.items():
            # Collect all data for this group
            prompt_str = None
            responses = []
            ground_truths = []
            data_sources = []
            extra_infos = []
            valid_response_lengths = []

            for idx in indices:
                data_item = data[idx]

                # Decode prompt (should be the same for all items in the group)
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                if prompt_str is None:
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

                # Decode response
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                # Collect metadata
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {}).copy()
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                responses.append(response_str)
                ground_truths.append(ground_truth)
                data_sources.append(data_source)
                extra_infos.append(extra_info)
                valid_response_lengths.append(valid_response_length)

            # Call the group-aware compute_score function
            scores = self.compute_score(
                prompt_str=prompt_str,
                responses=responses,
                ground_truths=ground_truths,
                data_sources=data_sources,
                extra_infos=extra_infos,
            )

            # Handle both list of floats and list of dicts
            if isinstance(scores, list):
                if len(scores) != len(indices):
                    raise ValueError(
                        f"compute_score returned {len(scores)} scores but expected {len(indices)} "
                        f"(one for each response in the group)"
                    )

                for i, (idx, score) in enumerate(zip(indices, scores)):
                    if isinstance(score, dict):
                        reward = score["score"]
                        # Store extra information
                        for key, value in score.items():
                            reward_extra_info[key].append(value)
                    else:
                        reward = score

                    # Assign reward to the last token of the response
                    valid_response_length = valid_response_lengths[i]
                    reward_tensor[idx, valid_response_length - 1] = reward
            else:
                raise ValueError(
                    f"compute_score must return a list of scores (one per response), "
                    f"but got {type(scores)}"
                )

            # Print debug information for the first few groups
            if examined_groups < self.num_examine:
                examined_groups += 1
                print(f"\n[Group UID: {uid}]")
                print(f"[prompt] {prompt_str}")
                print(f"[num_responses] {len(responses)}")
                for i, (response, score) in enumerate(zip(responses, scores)):
                    print(f"\n  [response_{i}] {response}")
                    print(f"  [ground_truth_{i}] {ground_truths[i]}")
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"  [{key}_{i}] {value}")
                    else:
                        print(f"  [score_{i}] {score}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
