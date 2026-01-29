#!/usr/bin/env python3
"""
Create sample DNA data for testing the Megatron-LM pipeline.

This script generates random DNA sequences in JSONL format, suitable for
testing the tokenizer training and data preprocessing pipeline.

Usage:
    python scripts/create_sample_data.py --output-dir processed_data --num-sequences 10000
"""

import argparse
import json
import os
import random
from pathlib import Path


def generate_dna_sequence(length: int = 1024) -> str:
    """Generate a random DNA sequence."""
    bases = "ACGT"
    # Occasionally add N (unknown base)
    if random.random() < 0.01:
        bases = "ACGTN"
    return "".join(random.choice(bases) for _ in range(length))


def generate_realistic_dna_sequence(length: int = 1024) -> str:
    """
    Generate a more realistic DNA sequence with some patterns.
    
    Real DNA sequences have:
    - GC content around 40-60%
    - Repetitive regions
    - AT-rich and GC-rich regions
    """
    sequence = []
    i = 0
    
    while i < length:
        # Decide what type of region to generate
        region_type = random.random()
        
        if region_type < 0.1:  # 10% chance of repeat region
            # Generate a short repeat
            repeat_unit = "".join(random.choices("ACGT", k=random.randint(2, 6)))
            repeat_count = random.randint(3, 10)
            region = (repeat_unit * repeat_count)[:min(50, length - i)]
            sequence.append(region)
            i += len(region)
            
        elif region_type < 0.2:  # 10% chance of AT-rich region
            region_len = min(random.randint(20, 100), length - i)
            region = "".join(random.choices("AT", weights=[0.5, 0.5], k=region_len))
            sequence.append(region)
            i += region_len
            
        elif region_type < 0.3:  # 10% chance of GC-rich region
            region_len = min(random.randint(20, 100), length - i)
            region = "".join(random.choices("GC", weights=[0.5, 0.5], k=region_len))
            sequence.append(region)
            i += region_len
            
        else:  # 70% normal balanced region
            region_len = min(random.randint(50, 200), length - i)
            # Balanced nucleotide distribution
            region = "".join(random.choices("ACGT", weights=[0.25, 0.25, 0.25, 0.25], k=region_len))
            sequence.append(region)
            i += region_len
    
    result = "".join(sequence)[:length]
    
    # Occasionally add N bases (unknown)
    if random.random() < 0.05:
        pos = random.randint(0, len(result) - 1)
        n_len = random.randint(1, 10)
        result = result[:pos] + "N" * n_len + result[pos + n_len:]
    
    return result[:length]


def create_jsonl_data(
    output_dir: str,
    num_sequences: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    realistic: bool = True,
):
    """
    Create JSONL data file with DNA sequences.
    
    Args:
        output_dir: Output directory
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        split: Dataset split name
        realistic: Use realistic DNA patterns
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}.jsonl")
    
    generator = generate_realistic_dna_sequence if realistic else generate_dna_sequence
    
    print(f"Generating {num_sequences:,} DNA sequences of length {sequence_length}...")
    
    with open(output_file, "w") as f:
        for i in range(num_sequences):
            sequence = generator(sequence_length)
            record = {
                "text": sequence,
                "chromosome": f"chr{random.randint(1, 22)}",
                "start_pos": random.randint(0, 250_000_000),
            }
            f.write(json.dumps(record) + "\n")
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} sequences...")
    
    print(f"Saved {num_sequences:,} sequences to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create sample DNA data for testing Megatron-LM pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_data",
        help="Output directory (default: ./processed_data)"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of sequences to generate (default: 10000)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Length of each sequence (default: 1024)"
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple random sequences instead of realistic patterns"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("Creating Sample DNA Data")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Total sequences: {args.num_sequences:,}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Realistic patterns: {not args.simple}")
    print("=" * 60)
    
    # Calculate split sizes
    val_size = int(args.num_sequences * args.validation_ratio)
    train_size = args.num_sequences - val_size
    
    # Create train data
    print(f"\nCreating train split ({train_size:,} sequences)...")
    create_jsonl_data(
        output_dir=args.output_dir,
        num_sequences=train_size,
        sequence_length=args.sequence_length,
        split="train",
        realistic=not args.simple,
    )
    
    # Create validation data
    if val_size > 0:
        print(f"\nCreating validation split ({val_size:,} sequences)...")
        create_jsonl_data(
            output_dir=args.output_dir,
            num_sequences=val_size,
            sequence_length=args.sequence_length,
            split="validation",
            realistic=not args.simple,
        )
    
    print("\n" + "=" * 60)
    print("Sample Data Creation Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    for f in Path(args.output_dir).glob("*.jsonl"):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f}: {size:.2f} MB")
    
    print(f"\nNext steps:")
    print(f"  1. Train tokenizer: python scripts/train_tokenizer.py --input {args.output_dir}/train.jsonl")
    print(f"  2. Convert to Megatron format using megatron_lm/tools/preprocess_data.py")


if __name__ == "__main__":
    main()
