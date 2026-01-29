"""
Data preparation script for DNA model training.

Processes local DNA sequence files (train.txt, val.txt) for Megatron-LM training.
Each line in the input files should be a DNA sequence (512bp) containing only A, C, T, G.

Usage:
    python data/prepare_data.py --input-dir data/pretrain --output-dir processed_data
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dna_tokenizer import DNATokenizer


def load_local_pretrain_data(
    split: str,
    data_dir: str = "data/pretrain",
    max_samples: Optional[int] = None
) -> Generator[Dict, None, None]:
    """
    Load DNA sequences from local train.txt or val.txt files.
    
    Args:
        split: Dataset split ('train' or 'validation')
        data_dir: Directory containing train.txt and val.txt
        max_samples: Maximum number of samples to load (None for all)
    
    Yields:
        Dictionary with 'text' key containing the DNA sequence
    """
    # Map split names to file names
    file_mapping = {
        'train': 'train.txt',
        'validation': 'val.txt',
        'val': 'val.txt',
        'test': 'val.txt',  # Use val.txt for test if no separate test file
    }
    
    filename = file_mapping.get(split, f'{split}.txt')
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading {split} data from: {filepath}")
    
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                yield {"text": seq}
                count += 1
                if max_samples and count >= max_samples:
                    break
    
    print(f"  Loaded {count:,} sequences")


def clean_sequence(sequence: str) -> str:
    """
    Clean a DNA sequence by converting to uppercase.
    Since input data only contains A, C, T, G, minimal cleaning needed.
    
    Args:
        sequence: Raw DNA sequence
    
    Returns:
        Cleaned uppercase sequence
    """
    return sequence.upper().strip()


def save_jsonl(
    output_dir: str,
    split: str = "train",
    input_dir: str = "data/pretrain",
    max_samples: Optional[int] = None
) -> Tuple[int, int]:
    """
    Save dataset as JSONL format for Megatron preprocessing.
    
    Args:
        output_dir: Output directory
        split: Dataset split
        input_dir: Input directory with train.txt/val.txt
        max_samples: Maximum samples
    
    Returns:
        Tuple of (num_sequences, num_tokens)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{split}.jsonl")
    
    total_sequences = 0
    total_tokens = 0
    
    with open(output_file, "w") as f:
        for item in tqdm(load_local_pretrain_data(split, input_dir, max_samples), 
                         desc=f"Processing {split}"):
            sequence = clean_sequence(item["text"])
            
            record = {"text": sequence}
            f.write(json.dumps(record) + "\n")
            total_sequences += 1
            total_tokens += len(sequence)
    
    print(f"\n{split} split saved to JSONL:")
    print(f"  Sequences: {total_sequences:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  File: {output_file}")
    
    return total_sequences, total_tokens


# Megatron index file format constants
# Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
_HDR_MAGIC = b'MMIDIDX\x00\x00'
_INDEX_VERSION = 1


def write_megatron_index(
    output_path: str,
    sizes: List[int],
    doc_indices: List[int],
    dtype: np.dtype = np.uint16,
):
    """
    Write Megatron-style index file.
    
    Format:
    - 9 bytes: magic number
    - 8 bytes: version (int64)
    - 1 byte: dtype code
    - 8 bytes: number of sequences (int64)
    - 8 bytes: number of documents (int64)
    - sizes array: int32 for each sequence
    - pointers array: int64 cumulative pointers
    - doc_indices array: int64 document start indices
    """
    # Dtype codes matching Megatron
    dtype_codes = {
        np.uint8: 1,
        np.int8: 2,
        np.int16: 3,
        np.int32: 4,
        np.int64: 5,
        np.float32: 6,
        np.float64: 7,
        np.uint16: 8,
    }
    
    dtype_code = dtype_codes.get(dtype, dtype_codes[np.uint16])
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(_HDR_MAGIC)
        f.write(struct.pack('<Q', _INDEX_VERSION))  # version
        f.write(struct.pack('B', dtype_code))  # dtype
        f.write(struct.pack('<Q', len(sizes)))  # num sequences
        f.write(struct.pack('<Q', len(doc_indices)))  # num documents
        
        # Write sizes
        sizes_array = np.array(sizes, dtype=np.int32)
        f.write(sizes_array.tobytes())
        
        # Write pointers (cumulative sum of sizes)
        pointers = np.zeros(len(sizes) + 1, dtype=np.int64)
        pointers[1:] = np.cumsum(sizes)
        f.write(pointers.tobytes())
        
        # Write document indices
        doc_idx_array = np.array(doc_indices, dtype=np.int64)
        f.write(doc_idx_array.tobytes())


def process_to_megatron_format(
    output_dir: str,
    split: str = "train",
    input_dir: str = "data/pretrain",
    max_samples: Optional[int] = None,
    append_eos: bool = True,
) -> Tuple[int, int]:
    """
    Process data directly to Megatron indexed binary format.
    
    Uses single nucleotide tokenization: A=7, T=8, C=9, G=10, EOS=6
    
    Args:
        output_dir: Output directory
        split: Dataset split
        input_dir: Input directory with train.txt/val.txt
        max_samples: Maximum samples
        append_eos: Whether to append EOS token to each sequence
    
    Returns:
        Tuple of (num_sequences, num_tokens)
    """
    # Single nucleotide tokenizer mapping (matches Megatron DNANucleotideTokenizer)
    NUCLEOTIDE_TO_ID = {'A': 7, 'T': 8, 'C': 9, 'G': 10}
    EOS_ID = 6
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files (Megatron format)
    bin_file = os.path.join(output_dir, f"{split}_text_document.bin")
    idx_file = os.path.join(output_dir, f"{split}_text_document.idx")
    
    sizes = []
    doc_indices = [0]
    total_tokens = 0
    
    print(f"Processing {split} to Megatron format...")
    print(f"  Output: {output_dir}/{split}_text_document.{{bin,idx}}")
    print(f"  Append EOS: {append_eos}")
    
    with open(bin_file, 'wb') as f_bin:
        for doc_idx, item in enumerate(tqdm(
            load_local_pretrain_data(split, input_dir, max_samples),
            desc=f"Tokenizing {split}"
        )):
            sequence = clean_sequence(item["text"])
            
            # Tokenize: convert nucleotides to IDs
            token_ids = [NUCLEOTIDE_TO_ID.get(c, 1) for c in sequence]  # 1 = UNK
            
            # Append EOS if requested
            if append_eos:
                token_ids.append(EOS_ID)
            
            # Write to binary file
            tokens_array = np.array(token_ids, dtype=np.uint16)
            f_bin.write(tokens_array.tobytes())
            
            # Track sizes
            sizes.append(len(token_ids))
            total_tokens += len(token_ids)
            
            # Each document is a separate entry
            doc_indices.append(len(sizes))
    
    # Remove the last doc_index (it's one past the end)
    doc_indices = doc_indices[:-1]
    
    # Write index file
    print(f"Writing index file...")
    write_megatron_index(idx_file, sizes, doc_indices, np.uint16)
    
    # Save metadata
    metadata = {
        "split": split,
        "num_sequences": len(sizes),
        "num_tokens": total_tokens,
        "vocab_size": 16,
        "dtype": "uint16",
        "append_eos": append_eos,
        "tokenizer": "DNANucleotideTokenizer",
    }
    
    meta_file = os.path.join(output_dir, f"{split}_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{split} split processed:")
    print(f"  Sequences: {len(sizes):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Avg seq length: {total_tokens / len(sizes):.1f}")
    
    return len(sizes), total_tokens


class DNADataset:
    """
    PyTorch-compatible dataset for loading preprocessed DNA data.
    
    Loads Megatron-format binary files.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory with preprocessed data
            split: Dataset split
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load metadata
        meta_file = os.path.join(data_dir, f"{split}_metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            self.num_sequences = self.metadata["num_sequences"]
        else:
            self.metadata = {}
            self.num_sequences = 0
        
        # Memory-map the binary file
        bin_file = os.path.join(data_dir, f"{split}_text_document.bin")
        if os.path.exists(bin_file):
            self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        else:
            self.data = None
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a single sequence (requires index file parsing)."""
        if self.data is None:
            raise ValueError("Data file not loaded")
        # Note: For proper indexing, need to parse the index file
        # This is a simplified version
        return self.data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DNA data for Megatron-LM training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/pretrain",
        help="Input directory containing train.txt and val.txt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to process"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per split (for testing)"
    )
    parser.add_argument(
        "--format",
        choices=["megatron", "jsonl", "both"],
        default="megatron",
        help="Output format (default: megatron)"
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Don't append EOS token to sequences"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DNA Data Preparation (Local Files)")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Format: {args.format}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 60)
    
    # Check input files exist
    for split in args.splits:
        filename = "train.txt" if split == "train" else "val.txt"
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
    
    total_sequences = 0
    total_tokens = 0
    
    # Create megatron subdirectory
    megatron_dir = os.path.join(args.output_dir, "megatron")
    
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        
        if args.format in ["megatron", "both"]:
            seq, tok = process_to_megatron_format(
                output_dir=megatron_dir,
                split=split,
                input_dir=args.input_dir,
                max_samples=args.max_samples,
                append_eos=not args.no_eos,
            )
            total_sequences += seq
            total_tokens += tok
        
        if args.format in ["jsonl", "both"]:
            seq, tok = save_jsonl(
                output_dir=args.output_dir,
                split=split,
                input_dir=args.input_dir,
                max_samples=args.max_samples
            )
            if args.format == "jsonl":
                total_sequences += seq
                total_tokens += tok
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total sequences: {total_sequences:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output: {args.output_dir}")
    
    if args.format in ["megatron", "both"]:
        print(f"\nMegatron data files:")
        print(f"  {megatron_dir}/train_text_document.bin")
        print(f"  {megatron_dir}/train_text_document.idx")
        print(f"\nTo train with Megatron-LM:")
        print(f"  --data-path {megatron_dir}/train_text_document")
        print(f"  --tokenizer-type DNANucleotideTokenizer")


if __name__ == "__main__":
    main()
