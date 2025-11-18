"""
Convert custom JSONL data to veRL training format for proactive agent.

This script converts JSONL data into the format required by veRL training pipeline.

Expected input JSONL format (one JSON object per line):
{
    "question": "What is 2+2?",
    "answer": "4",
    // ... other fields
}

Expected output format:
{
    "data_source": "custom_dataset",
    "prompt": [{"role": "user", "content": "question text"}],
    "ability": "reasoning",
    "reward_model": {"style": "rule", "ground_truth": "answer"},
    "extra_info": {"index": 0, ...}
}
"""

import argparse
import json
import os
from pathlib import Path

import datasets


def convert_jsonl_to_verl_format(
    input_file: str,
    output_dir: str,
    data_source_name: str = "proactive_dataset",
    ability: str = "reasoning",
    system_prompt: str = None,
    question_field: str = "question",
    answer_field: str = "answer",
):
    """
    Convert JSONL data to veRL format.

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save output parquet files
        data_source_name: Name of the data source
        ability: Ability category (e.g., "reasoning", "math", "coding")
        system_prompt: Optional system prompt to prepend to questions
        question_field: Field name for questions in input JSONL
        answer_field: Field name for answers in input JSONL
    """

    # Read JSONL file
    print(f"Reading data from {input_file}...")
    data_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_items.append(json.loads(line))

    print(f"Loaded {len(data_items)} items")

    # Convert to veRL format
    converted_data = []
    for idx, item in enumerate(data_items):
        # Extract question and answer
        question = item.get(question_field, "")
        answer = item.get(answer_field, "")

        if not question or not answer:
            print(f"Warning: Skipping item {idx} due to missing question or answer")
            continue

        # Build prompt
        if system_prompt:
            prompt_content = f"{system_prompt}\n\n{question}"
        else:
            prompt_content = question

        # Create veRL format
        verl_item = {
            "data_source": data_source_name,
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer)
            },
            "extra_info": {
                "index": idx,
                "original_data": item  # Keep original data for reference
            }
        }

        converted_data.append(verl_item)

    print(f"Converted {len(converted_data)} items")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as HuggingFace dataset and export to parquet
    dataset = datasets.Dataset.from_list(converted_data)

    # Split into train/test if needed
    # For now, save all as train
    output_path = os.path.join(output_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved dataset to {output_path}")

    # Save one example as JSON for reference
    example_path = os.path.join(output_dir, "train_example.json")
    with open(example_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data[0], f, indent=2, ensure_ascii=False)
    print(f"Saved example to {example_path}")

    print(f"\nConversion complete! Dataset saved to {output_dir}")
    print(f"Total items: {len(converted_data)}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL data to veRL format")

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/proactive_dataset",
        help="Output directory for converted data"
    )

    parser.add_argument(
        "--data_source_name",
        type=str,
        default="proactive_dataset",
        help="Name of the data source"
    )

    parser.add_argument(
        "--ability",
        type=str,
        default="reasoning",
        help="Ability category (e.g., reasoning, math, coding)"
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend to questions"
    )

    parser.add_argument(
        "--question_field",
        type=str,
        default="question",
        help="Field name for questions in input JSONL"
    )

    parser.add_argument(
        "--answer_field",
        type=str,
        default="answer",
        help="Field name for answers in input JSONL"
    )

    args = parser.parse_args()

    # Convert data
    convert_jsonl_to_verl_format(
        input_file=args.input_file,
        output_dir=args.output_dir,
        data_source_name=args.data_source_name,
        ability=args.ability,
        system_prompt=args.system_prompt,
        question_field=args.question_field,
        answer_field=args.answer_field,
    )


if __name__ == "__main__":
    main()
