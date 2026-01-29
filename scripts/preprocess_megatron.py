#!/usr/bin/env python3
"""
Preprocess DNA data into Megatron-LM indexed binary format.

This script supports two tokenization modes:
1. sentencepiece: SentencePiece BPE tokenizer (DEFAULT, matches Genos) - requires sentencepiece library
2. dna-nucleotide: Single nucleotide tokenizer (A=7, T=8, C=9, G=10) - NO dependencies

The output format matches Megatron's expected format:
- {prefix}_text_document.bin: Binary file with tokenized data
- {prefix}_text_document.idx: Index file with document offsets

Usage (SentencePiece - default, recommended):
    python scripts/preprocess_megatron.py \
        --input data/pretrain/train.txt \
        --output-prefix processed_data/megatron/train

    # With custom tokenizer model:
    python scripts/preprocess_megatron.py \
        --input data/pretrain/train.txt \
        --output-prefix processed_data/megatron/train \
        --tokenizer-model ./my_tokenizer.model

Usage (DNA nucleotide - alternative):
    python scripts/preprocess_megatron.py \
        --input data/pretrain/train.txt \
        --output-prefix processed_data/megatron/train \
        --tokenizer-type dna-nucleotide
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Iterator, Dict

import numpy as np


# Megatron index file format constants
# Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
_HDR_MAGIC = b'MMIDIDX\x00\x00'
_INDEX_VERSION = 1


# DNA Nucleotide Tokenizer vocabulary (matches Megatron DNANucleotideTokenizer)
DNA_VOCAB = {
    '[PAD]': 0,
    '[UNK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    '[MASK]': 4,
    '[BOS]': 5,
    '[EOS]': 6,
    'A': 7,
    'T': 8,
    'C': 9,
    'G': 10,
}
DNA_NUCLEOTIDES = {'A': 7, 'T': 8, 'C': 9, 'G': 10}
DNA_EOS_ID = 6
DNA_UNK_ID = 1
DNA_VOCAB_SIZE = 16


class DNANucleotideTokenizer:
    """Simple single nucleotide tokenizer for DNA sequences."""
    
    def __init__(self):
        self.vocab = DNA_VOCAB
        self.nucleotides = DNA_NUCLEOTIDES
        self.eos_id = DNA_EOS_ID
        self.unk_id = DNA_UNK_ID
        self.vocab_size = DNA_VOCAB_SIZE
    
    def encode(self, text: str) -> List[int]:
        """Encode DNA sequence to token IDs."""
        ids = []
        for char in text.upper():
            if char in self.nucleotides:
                ids.append(self.nucleotides[char])
            elif char in ' \n\t':
                continue  # Skip whitespace
            else:
                ids.append(self.unk_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to DNA sequence."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        chars = []
        for i in ids:
            if i in self.nucleotides.values():
                chars.append(inv_vocab.get(i, ''))
        return ''.join(chars)


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
    - sizes array: int32 for each sequence (N elements)
    - pointers array: int64 byte offsets to each sequence (N elements)
    - doc_indices array: int64 document start indices (num_documents elements)
    
    Note: Megatron reads exactly N pointers (one per sequence), where each pointer
    is the byte offset to the start of that sequence in the .bin file.
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
        
        # Write pointers (byte offsets to the start of each sequence)
        # Megatron expects exactly len(sizes) pointers, NOT len(sizes)+1
        # Each pointer is a byte offset = cumulative tokens * dtype_size
        dtype_size = np.dtype(dtype).itemsize
        pointers = np.zeros(len(sizes), dtype=np.int64)
        if len(sizes) > 1:
            pointers[1:] = np.cumsum(sizes[:-1]) * dtype_size
        f.write(pointers.tobytes())
        
        # Write document indices
        doc_idx_array = np.array(doc_indices, dtype=np.int64)
        f.write(doc_idx_array.tobytes())


def read_input_file(input_path: str, input_format: str, json_key: str = "text") -> Iterator[str]:
    """
    Read input file and yield text sequences.
    
    Args:
        input_path: Path to input file
        input_format: 'txt' for plain text (one sequence per line), 'jsonl' for JSON lines
        json_key: Key to extract text from JSON objects
    
    Yields:
        Text sequences
    """
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if input_format == 'jsonl':
                try:
                    obj = json.loads(line)
                    text = obj.get(json_key, "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
            else:  # txt format
                yield line


def preprocess_dna_nucleotide(
    input_path: str,
    output_prefix: str,
    input_format: str = "txt",
    json_key: str = "text",
    append_eod: bool = True,
    dtype: str = "uint16",
    log_interval: int = 10000,
):
    """
    Convert input file to Megatron indexed binary format using DNA nucleotide tokenizer.
    
    Args:
        input_path: Path to input file (txt or jsonl)
        output_prefix: Output prefix (will create {prefix}_text_document.bin/.idx)
        input_format: Input format ('txt' or 'jsonl')
        json_key: JSON key containing text (for jsonl format)
        append_eod: Whether to append EOD token to each document
        dtype: Data type for token IDs
        log_interval: Log progress every N documents
    """
    tokenizer = DNANucleotideTokenizer()
    
    # Determine output dtype
    np_dtype = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'int16': np.int16,
        'int32': np.int32,
    }.get(dtype, np.uint16)
    
    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    bin_file = f"{output_prefix}_text_document.bin"
    idx_file = f"{output_prefix}_text_document.idx"
    
    # Process documents
    sizes = []
    doc_indices = [0]
    total_tokens = 0
    
    print(f"Processing {input_path}...")
    print(f"  Tokenizer: DNANucleotideTokenizer (vocab_size={tokenizer.vocab_size})")
    print(f"  Input format: {input_format}")
    print(f"  Output: {output_prefix}_text_document.{{bin,idx}}")
    print(f"  Dtype: {dtype}")
    print(f"  Append EOD: {append_eod} (EOD id={tokenizer.eos_id})")
    
    with open(bin_file, 'wb') as f_bin:
        for doc_idx, text in enumerate(read_input_file(input_path, input_format, json_key)):
            # Tokenize
            token_ids = tokenizer.encode(text)
            
            # Append EOD if requested
            if append_eod:
                token_ids.append(tokenizer.eos_id)
            
            # Write to binary file
            tokens_array = np.array(token_ids, dtype=np_dtype)
            f_bin.write(tokens_array.tobytes())
            
            # Track sizes
            sizes.append(len(token_ids))
            total_tokens += len(token_ids)
            
            # Each document is a separate entry
            doc_indices.append(len(sizes))
            
            if (doc_idx + 1) % log_interval == 0:
                print(f"  Processed {doc_idx + 1:,} documents, {total_tokens:,} tokens")
    
    # NOTE: doc_indices should have len(sizes)+1 elements
    # doc_indices[i] = starting sequence index of document i
    # doc_indices[-1] = total number of sequences (required by Megatron)
    
    # Write index file
    print(f"Writing index file...")
    write_megatron_index(idx_file, sizes, doc_indices, np_dtype)
    
    print(f"\nPreprocessing complete!")
    print(f"  Documents: {len(doc_indices) - 1:,}")
    print(f"  Sequences: {len(sizes):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Avg tokens/doc: {total_tokens / len(sizes):.1f}")
    print(f"  Output: {bin_file}")
    print(f"  Index: {idx_file}")
    
    # Verify files
    bin_size = os.path.getsize(bin_file)
    idx_size = os.path.getsize(idx_file)
    print(f"  Binary size: {bin_size / (1024*1024):.2f} MB")
    print(f"  Index size: {idx_size / 1024:.2f} KB")
    
    # Save metadata
    metadata = {
        "input_file": input_path,
        "num_documents": len(doc_indices) - 1,
        "num_sequences": len(sizes),
        "num_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype,
        "append_eod": append_eod,
        "tokenizer_type": "DNANucleotideTokenizer",
    }
    meta_file = f"{output_prefix}_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_file}")


def preprocess_sentencepiece(
    input_path: str,
    output_prefix: str,
    tokenizer_model: str,
    input_format: str = "txt",
    json_key: str = "text",
    append_eod: bool = True,
    dtype: str = "uint16",
    log_interval: int = 10000,
):
    """
    Convert input file to Megatron indexed binary format using SentencePiece tokenizer.
    
    Args:
        input_path: Path to input file (txt or jsonl)
        output_prefix: Output prefix (will create {prefix}_text_document.bin/.idx)
        tokenizer_model: Path to SentencePiece model
        input_format: Input format ('txt' or 'jsonl')
        json_key: JSON key containing text (for jsonl format)
        append_eod: Whether to append EOD token to each document
        dtype: Data type for token IDs
        log_interval: Log progress every N documents
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("Error: sentencepiece not installed. Run: pip install sentencepiece")
        print("Or use --tokenizer-type dna-nucleotide for DNA sequences (no dependencies)")
        sys.exit(1)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)
    
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()
    
    # Determine output dtype
    np_dtype = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'int16': np.int16,
        'int32': np.int32,
    }.get(dtype, np.uint16)
    
    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    bin_file = f"{output_prefix}_text_document.bin"
    idx_file = f"{output_prefix}_text_document.idx"
    
    # Process documents
    sizes = []
    doc_indices = [0]
    total_tokens = 0
    
    print(f"Processing {input_path}...")
    print(f"  Tokenizer: {tokenizer_model} (vocab_size={vocab_size})")
    print(f"  Input format: {input_format}")
    print(f"  Output: {output_prefix}_text_document.{{bin,idx}}")
    print(f"  Dtype: {dtype}")
    print(f"  Append EOD: {append_eod} (EOD id={eos_id})")
    
    with open(bin_file, 'wb') as f_bin:
        for doc_idx, text in enumerate(read_input_file(input_path, input_format, json_key)):
            # Tokenize
            token_ids = sp.encode_as_ids(text)
            
            # Append EOD if requested
            if append_eod and eos_id is not None:
                token_ids.append(eos_id)
            
            # Write to binary file
            tokens_array = np.array(token_ids, dtype=np_dtype)
            f_bin.write(tokens_array.tobytes())
            
            # Track sizes
            sizes.append(len(token_ids))
            total_tokens += len(token_ids)
            
            # Each document is a separate entry
            doc_indices.append(len(sizes))
            
            if (doc_idx + 1) % log_interval == 0:
                print(f"  Processed {doc_idx + 1:,} documents, {total_tokens:,} tokens")
    
    # NOTE: doc_indices should have len(sizes)+1 elements
    # doc_indices[i] = starting sequence index of document i
    # doc_indices[-1] = total number of sequences (required by Megatron)
    
    # Write index file
    print(f"Writing index file...")
    write_megatron_index(idx_file, sizes, doc_indices, np_dtype)
    
    print(f"\nPreprocessing complete!")
    print(f"  Documents: {len(doc_indices) - 1:,}")
    print(f"  Sequences: {len(sizes):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Avg tokens/doc: {total_tokens / len(sizes):.1f}")
    print(f"  Output: {bin_file}")
    print(f"  Index: {idx_file}")
    
    # Verify files
    bin_size = os.path.getsize(bin_file)
    idx_size = os.path.getsize(idx_file)
    print(f"  Binary size: {bin_size / (1024*1024):.2f} MB")
    print(f"  Index size: {idx_size / 1024:.2f} KB")
    
    # Save metadata
    metadata = {
        "input_file": input_path,
        "num_documents": len(doc_indices) - 1,
        "num_sequences": len(sizes),
        "num_tokens": total_tokens,
        "vocab_size": vocab_size,
        "dtype": dtype,
        "append_eod": append_eod,
        "tokenizer_type": "SentencePieceTokenizer",
        "tokenizer_model": tokenizer_model,
    }
    meta_file = f"{output_prefix}_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_file}")


def verify_dna_output(output_prefix: str, num_samples: int = 3):
    """Verify the created files can be read correctly (DNA tokenizer)."""
    bin_file = f"{output_prefix}_text_document.bin"
    idx_file = f"{output_prefix}_text_document.idx"
    
    tokenizer = DNANucleotideTokenizer()
    
    print(f"\n=== Verification (DNA Nucleotide Tokenizer) ===")
    
    # Read index header
    with open(idx_file, 'rb') as f:
        magic = f.read(9)
        version = struct.unpack('<Q', f.read(8))[0]
        dtype_code = struct.unpack('B', f.read(1))[0]
        num_seqs = struct.unpack('<Q', f.read(8))[0]
        num_docs = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Index header:")
        print(f"  Magic: {magic}")
        print(f"  Version: {version}")
        print(f"  Dtype code: {dtype_code}")
        print(f"  Sequences: {num_seqs:,}")
        print(f"  Documents: {num_docs:,}")
        
        # Read sizes (token counts per sequence)
        sizes = np.frombuffer(f.read(num_seqs * 4), dtype=np.int32)
        
        # Read pointers (byte offsets, N elements not N+1)
        pointers = np.frombuffer(f.read(num_seqs * 8), dtype=np.int64)
    
    # Read and decode a few samples
    dtype_map = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 8: np.uint16}
    np_dtype = dtype_map.get(dtype_code, np.uint16)
    dtype_size = np.dtype(np_dtype).itemsize
    
    with open(bin_file, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np_dtype)
    
    print(f"\nSample documents:")
    for i in range(min(num_samples, num_seqs)):
        # pointers are byte offsets, convert to token index
        start_token = pointers[i] // dtype_size
        num_tokens = sizes[i]
        tokens = data[start_token:start_token + num_tokens].tolist()
        text = tokenizer.decode(tokens)
        print(f"  Doc {i}: {len(tokens)} tokens -> '{text[:60]}{'...' if len(text) > 60 else ''}'")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DNA data into Megatron-LM indexed format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (txt for plain text, jsonl for JSON lines)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix (creates {prefix}_text_document.bin/.idx)"
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="sentencepiece",
        choices=["sentencepiece", "dna-nucleotide"],
        help="Tokenizer type (default: sentencepiece)"
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="./tokenizer/dna_tokenizer.model",
        help="Path to SentencePiece model file (default: ./tokenizer/dna_tokenizer.model)"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default=None,
        choices=["txt", "jsonl"],
        help="Input format (auto-detected from extension if not specified)"
    )
    parser.add_argument(
        "--json-keys",
        type=str,
        default="text",
        help="JSON key containing text (default: text)"
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        default=True,
        help="Append EOD token to each document (default: True)"
    )
    parser.add_argument(
        "--no-append-eod",
        action="store_true",
        help="Don't append EOD token"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "int16", "int32"],
        help="Data type for token IDs (default: uint16)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Log interval for progress (default: 10000)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output after preprocessing"
    )
    
    args = parser.parse_args()
    
    # Auto-detect input format
    if args.input_format is None:
        if args.input.endswith('.jsonl') or args.input.endswith('.json'):
            args.input_format = 'jsonl'
        else:
            args.input_format = 'txt'
    
    append_eod = not args.no_append_eod
    
    if args.tokenizer_type == "sentencepiece":
        # Check tokenizer model exists
        if not os.path.exists(args.tokenizer_model):
            print(f"Error: SentencePiece model not found at {args.tokenizer_model}")
            print("")
            print("Train a tokenizer first:")
            print("  python scripts/train_tokenizer.py \\")
            print("    --input data/pretrain/train.txt \\")
            print("    --output-prefix tokenizer/dna_tokenizer \\")
            print("    --vocab-size 128")
            print("")
            print("Or use the nucleotide tokenizer (no model needed):")
            print("  python scripts/preprocess_megatron.py --tokenizer-type dna-nucleotide ...")
            sys.exit(1)
        
        preprocess_sentencepiece(
            input_path=args.input,
            output_prefix=args.output_prefix,
            tokenizer_model=args.tokenizer_model,
            input_format=args.input_format,
            json_key=args.json_keys,
            append_eod=append_eod,
            dtype=args.dtype,
            log_interval=args.log_interval,
        )
    
    elif args.tokenizer_type == "dna-nucleotide":
        preprocess_dna_nucleotide(
            input_path=args.input,
            output_prefix=args.output_prefix,
            input_format=args.input_format,
            json_key=args.json_keys,
            append_eod=append_eod,
            dtype=args.dtype,
            log_interval=args.log_interval,
        )
        
        if args.verify:
            verify_dna_output(args.output_prefix)


if __name__ == "__main__":
    main()
