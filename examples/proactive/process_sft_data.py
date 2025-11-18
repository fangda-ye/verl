"""
Process SFT data from sampleQA_processed_2.jsonl for supervised fine-tuning.

This script randomly samples N items and converts them to SFT format.

Input format (sampleQA_processed_2.jsonl):
{
  "messages": [
    {"role": "system", "content": "You are a helpful proactive assistant."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "<think>...</think>\n<proactive>...</proactive>\nAnswer"}
  ],
  "id": 5,
  "sub_category": "simpleQA"
}

Output format (for SFT):
{
  "messages": [same as input],
  "id": 5,
  "sub_category": "simpleQA"
}
"""

import argparse
import json
import os
import random


def process_sft_data(
    input_file: str,
    output_file: str,
    num_samples: int = 50,
    seed: int = 42,
):
    """
    Process SFT data by randomly sampling N items.

    Args:
        input_file: Path to sampleQA_processed_2.jsonl
        output_file: Path to output JSONL file
        num_samples: Number of samples to randomly select
        seed: Random seed for reproducibility
    """

    print(f"Processing SFT data from {input_file}...")

    # Set random seed
    random.seed(seed)

    # Read all data
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

    # Randomly sample
    if num_samples > len(data_items):
        print(f"Warning: Requested {num_samples} samples but only {len(data_items)} available")
        num_samples = len(data_items)
        sampled_items = data_items
    else:
        sampled_items = random.sample(data_items, num_samples)

    print(f"Randomly sampled {len(sampled_items)} items")

    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ SFT data saved to {output_file}")
    print(f"  Total samples: {len(sampled_items)}")
    print(f"  Random seed: {seed}")

    # Print example
    if sampled_items:
        print("\n" + "="*80)
        print("Example item:")
        print("="*80)
        print(json.dumps(sampled_items[0], indent=2, ensure_ascii=False))
        print("="*80)

    return sampled_items


def main():
    parser = argparse.ArgumentParser(
        description="Process SFT data from sampleQA_processed_2.jsonl"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/sampleQA_processed_2.jsonl",
        help="Path to input sampleQA_processed_2.jsonl file"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/sft_samples.jsonl",
        help="Path to output JSONL file"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to randomly select (default: 50)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return 1

    # Process data
    try:
        process_sft_data(
            input_file=args.input_file,
            output_file=args.output_file,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        return 0
    except Exception as e:
        print(f"\nError processing data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
