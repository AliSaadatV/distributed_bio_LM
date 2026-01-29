#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer on DNA sequences for use with Megatron-LM.

This script trains a BPE-based SentencePiece model on DNA data, which learns
subword patterns in DNA sequences (e.g., "ATCG", "AAAA", common motifs).

Usage:
    python scripts/train_tokenizer.py \
        --input processed_data/train.jsonl \
        --output-prefix tokenizer/dna_tokenizer \
        --vocab-size 4096
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)


def extract_text_from_jsonl(jsonl_path: str, output_path: str, text_key: str = "text") -> int:
    """
    Extract text from JSONL file to plain text file for SentencePiece training.
    
    Args:
        jsonl_path: Path to JSONL file
        output_path: Path to output text file
        text_key: Key in JSON objects containing the text
    
    Returns:
        Number of sequences extracted
    """
    count = 0
    with open(jsonl_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if line.strip():
                try:
                    obj = json.loads(line)
                    text = obj.get(text_key, "")
                    if text:
                        f_out.write(text + "\n")
                        count += 1
                except json.JSONDecodeError:
                    continue
    return count


def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 4096,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    max_sentence_length: int = 16384,
    num_threads: int = 8,
    train_extremely_large_corpus: bool = False,
    input_sentence_size: int = 0,
):
    """
    Train a SentencePiece model on input data.
    
    Args:
        input_file: Path to input text file (one sequence per line)
        model_prefix: Output model prefix (produces {prefix}.model and {prefix}.vocab)
        vocab_size: Target vocabulary size
        model_type: Model type ('bpe', 'unigram', 'char', or 'word')
        character_coverage: Character coverage for training
        max_sentence_length: Maximum sentence length
        num_threads: Number of threads for training
        train_extremely_large_corpus: Enable for very large datasets
        input_sentence_size: Number of sentences to sample (0 for all)
    """
    # Create output directory if needed
    output_dir = os.path.dirname(model_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Build training command
    train_args = {
        "input": input_file,
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "max_sentence_length": max_sentence_length,
        "num_threads": num_threads,
        # Special token IDs matching Megatron-LM expectations
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        # Additional settings for DNA
        "split_by_whitespace": True,
        "split_by_number": False,
        "split_digits": False,
        "byte_fallback": False,
        "allow_whitespace_only_pieces": False,
        "remove_extra_whitespaces": True,
        "normalization_rule_name": "identity",  # Don't normalize DNA sequences
    }
    
    if train_extremely_large_corpus:
        train_args["train_extremely_large_corpus"] = True
    
    if input_sentence_size > 0:
        train_args["input_sentence_size"] = input_sentence_size
        train_args["shuffle_input_sentence"] = True
    
    print(f"Training SentencePiece model...")
    print(f"  Input: {input_file}")
    print(f"  Output: {model_prefix}.model, {model_prefix}.vocab")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Character coverage: {character_coverage}")
    
    spm.SentencePieceTrainer.train(**train_args)
    
    print(f"\nTraining complete!")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab: {model_prefix}.vocab")


def verify_tokenizer(model_path: str, test_sequences: list = None):
    """
    Verify the trained tokenizer works correctly.
    
    Args:
        model_path: Path to .model file
        test_sequences: Optional list of sequences to test
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"\n=== Tokenizer Verification ===")
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"Pad ID: {sp.pad_id()}")
    print(f"Unk ID: {sp.unk_id()}")
    print(f"BOS ID: {sp.bos_id()}")
    print(f"EOS ID: {sp.eos_id()}")
    
    # Default test sequences if not provided
    if test_sequences is None:
        test_sequences = [
            "ATCGATCGATCGATCG",
            "AAAAAAAAAAAAAAAAA",
            "GCGCGCGCGCGCGCGC",
            "ATATATATATATATAT",
            "NNNNNNNNNNNNNNN",
            "ATCGNNNNATCGATCG",
        ]
    
    print(f"\n=== Sample Tokenizations ===")
    for seq in test_sequences:
        tokens = sp.encode_as_pieces(seq)
        ids = sp.encode_as_ids(seq)
        print(f"  {seq[:30]:30} -> {len(tokens):3} tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
    
    # Verify roundtrip
    print(f"\n=== Roundtrip Verification ===")
    for seq in test_sequences[:3]:
        encoded = sp.encode_as_ids(seq)
        decoded = sp.decode_ids(encoded)
        match = "OK" if decoded == seq else "MISMATCH"
        print(f"  {seq[:20]:20} -> encode -> decode: {match}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer on DNA sequences for Megatron-LM"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (JSONL or plain text with one sequence per line)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="tokenizer/dna_tokenizer",
        help="Output model prefix (default: tokenizer/dna_tokenizer)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size (default: 4096)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "char"],
        help="SentencePiece model type (default: bpe)"
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="JSON key containing text in JSONL files (default: text)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of training threads (default: 8)"
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=16384,
        help="Maximum sentence length (default: 16384)"
    )
    parser.add_argument(
        "--large-corpus",
        action="store_true",
        help="Enable training on extremely large corpus"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of sentences to sample for training (0 for all)"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip tokenizer verification after training"
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    temp_file = None
    
    # Check if input is JSONL and needs extraction
    if input_path.endswith(".jsonl"):
        print(f"Extracting text from JSONL file: {input_path}")
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        temp_file.close()
        
        count = extract_text_from_jsonl(input_path, temp_file.name, args.text_key)
        print(f"Extracted {count:,} sequences")
        input_path = temp_file.name
    
    try:
        # Train the tokenizer
        train_sentencepiece(
            input_file=input_path,
            model_prefix=args.output_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            num_threads=args.num_threads,
            max_sentence_length=args.max_sentence_length,
            train_extremely_large_corpus=args.large_corpus,
            input_sentence_size=args.sample_size,
        )
        
        # Verify the tokenizer
        if not args.skip_verify:
            model_path = f"{args.output_prefix}.model"
            verify_tokenizer(model_path)
        
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    print(f"\n=== Done ===")
    print(f"Tokenizer files created:")
    print(f"  {args.output_prefix}.model  (use with --tokenizer-model in Megatron)")
    print(f"  {args.output_prefix}.vocab  (vocabulary file)")


if __name__ == "__main__":
    main()
