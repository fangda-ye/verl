"""
Process sampleQA.jsonl data for proactive agent training.

This script converts the sampleQA JSONL format to veRL training format.

Input format:
{
  "id": 0,
  "messages": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"}
  ],
  "answer": {...},  # Not used
  "sub_category": "simpleQA"
}

Output format:
{
  "data_source": "sampleQA",
  "prompt": [
    {"role": "system", "content": "You are a helpful proactive assistant."},
    {"role": "user", "content": "question"}
  ],
  "ability": "reasoning",
  "reward_model": {"style": "rule", "ground_truth": "answer"},
  "extra_info": {"index": 0, "sub_category": "simpleQA"}
}
"""

import argparse
import json
import os
from pathlib import Path

import datasets


def process_sampleQA(
    input_file: str,
    output_dir: str,
    system_prompt: str = "You are a helpful proactive assistant.",
    split_ratio: float = 0.95,
):
    """
    Process sampleQA JSONL data to veRL format.

    Args:
        input_file: Path to sampleQA.jsonl file
        output_dir: Directory to save output parquet files
        system_prompt: System prompt to add to all conversations
        split_ratio: Ratio for train/test split (default: 0.95 for train)
    """

    print(f"Processing sampleQA data from {input_file}...")

    # Read JSONL file
    data_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data_items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                    continue

    print(f"Loaded {len(data_items)} items")

    # Convert to veRL format
    converted_data = []
    skipped = 0

    for idx, item in enumerate(data_items):
        try:
            # Extract messages
            messages = item.get("messages", [])

            if len(messages) < 2:
                print(f"Warning: Skipping item {idx} - not enough messages")
                skipped += 1
                continue

            # Get user question and assistant answer
            user_message = None
            assistant_message = None

            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "").strip()
                elif msg.get("role") == "assistant":
                    assistant_message = msg.get("content", "").strip()

            if not user_message or not assistant_message:
                print(f"Warning: Skipping item {idx} - missing user question or assistant answer")
                skipped += 1
                continue

            # Build prompt with system message
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Get metadata
            item_id = item.get("id", idx)
            sub_category = item.get("sub_category", "unknown")

            # Create veRL format
            verl_item = {
                "data_source": "sampleQA",
                "prompt": prompt,
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": assistant_message
                },
                "extra_info": {
                    "index": idx,
                    "original_id": item_id,
                    "sub_category": sub_category,
                }
            }

            converted_data.append(verl_item)

        except Exception as e:
            print(f"Warning: Skipping item {idx} due to error: {e}")
            skipped += 1
            continue

    print(f"Successfully converted {len(converted_data)} items")
    if skipped > 0:
        print(f"Skipped {skipped} items due to errors")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split into train and test
    total_items = len(converted_data)
    train_size = int(total_items * split_ratio)

    train_data = converted_data[:train_size]
    test_data = converted_data[train_size:]

    print(f"\nSplitting data:")
    print(f"  Train: {len(train_data)} items ({split_ratio*100:.1f}%)")
    print(f"  Test: {len(test_data)} items ({(1-split_ratio)*100:.1f}%)")

    # Save as parquet using HuggingFace datasets
    if train_data:
        train_dataset = datasets.Dataset.from_list(train_data)
        train_path = os.path.join(output_dir, "train.parquet")
        train_dataset.to_parquet(train_path)
        print(f"\nSaved training data to {train_path}")

        # Save one example as JSON for reference
        example_path = os.path.join(output_dir, "train_example.json")
        with open(example_path, 'w', encoding='utf-8') as f:
            json.dump(train_data[0], f, indent=2, ensure_ascii=False)
        print(f"Saved example to {example_path}")

    if test_data:
        test_dataset = datasets.Dataset.from_list(test_data)
        test_path = os.path.join(output_dir, "test.parquet")
        test_dataset.to_parquet(test_path)
        print(f"Saved test data to {test_path}")
    else:
        print("\nWarning: No test data (split_ratio is too high or dataset is too small)")
        print("Creating a small test set from train data for validation...")
        # Create a small test set from the last few items of train
        test_size = min(10, len(train_data) // 10)  # 10% or 10 items, whichever is smaller
        test_data = train_data[-test_size:]
        test_dataset = datasets.Dataset.from_list(test_data)
        test_path = os.path.join(output_dir, "test.parquet")
        test_dataset.to_parquet(test_path)
        print(f"Created test set with {len(test_data)} items")

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total items processed: {len(converted_data)}")
    print(f"  System prompt: '{system_prompt}'")

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description="Process sampleQA.jsonl data to veRL format"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/sampleQA.jsonl",
        help="Path to input sampleQA.jsonl file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_sampleQA",
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful proactive assistant.",
        help="System prompt to add to all conversations"
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.95,
        help="Train/test split ratio (default: 0.95)"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        print(f"\nPlease make sure the sampleQA.jsonl file exists at: {args.input_file}")
        return 1

    # Process data
    try:
        process_sampleQA(
            input_file=args.input_file,
            output_dir=args.output_dir,
            system_prompt=args.system_prompt,
            split_ratio=args.split_ratio,
        )
        return 0
    except Exception as e:
        print(f"\nError processing data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
