#!/bin/bash
# Data Preparation Pipeline for Megatron-LM Training
#
# This script prepares DNA data for Megatron-LM training.
# Default tokenizer is SentencePiece (BPE), matching Genos approach.
#
# Input: data/pretrain/train.txt and data/pretrain/val.txt
#   - Each line is a 512bp DNA sequence containing only A, C, T, G
#
# Output: Megatron-compatible binary files
#   - processed_data/megatron/train_text_document.bin
#   - processed_data/megatron/train_text_document.idx
#   - processed_data/megatron/validation_text_document.bin
#   - processed_data/megatron/validation_text_document.idx
#
# Default Tokenizer: SentencePieceTokenizer (BPE, 128 vocab recommended)
#   - Train with: python scripts/train_tokenizer.py --vocab-size 128
#
# Alternative Tokenizer (--nucleotide-tokenizer): DNANucleotideTokenizer (vocab_size=16)
#   - Special tokens: PAD(0), UNK(1), CLS(2), SEP(3), MASK(4), BOS(5), EOS(6)
#   - Nucleotides: A(7), T(8), C(9), G(10)
#   - Reserved: 11-15
#
# Usage:
#   ./scripts/prepare_megatron_data.sh [OPTIONS]
#
# Example:
#   ./scripts/prepare_megatron_data.sh                              # SentencePiece (default)
#   ./scripts/prepare_megatron_data.sh --tokenizer-model ./my.model # Custom model
#   ./scripts/prepare_megatron_data.sh --nucleotide-tokenizer       # Use nucleotide tokenizer
#   ./scripts/prepare_megatron_data.sh --verify                     # With verification

set -e  # Exit on error

# Default values
INPUT_DIR="./data/pretrain"
OUTPUT_DIR="./processed_data"
VERIFY=false

# Tokenizer settings (default: SentencePiece)
USE_NUCLEOTIDE_TOKENIZER=false
TOKENIZER_MODEL="./tokenizer/dna_tokenizer.model"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --tokenizer-model)
            TOKENIZER_MODEL=$2
            shift 2
            ;;
        --nucleotide-tokenizer)
            USE_NUCLEOTIDE_TOKENIZER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir D           Input directory with train.txt/val.txt (default: ./data/pretrain)"
            echo "  --output-dir D          Output directory (default: ./processed_data)"
            echo "  --tokenizer-model PATH  Path to SentencePiece model (default: ./tokenizer/dna_tokenizer.model)"
            echo "  --nucleotide-tokenizer  Use DNANucleotideTokenizer instead of SentencePiece"
            echo "  --verify                Verify output after preprocessing"
            echo ""
            echo "Input files expected:"
            echo "  - \$INPUT_DIR/train.txt  (training sequences, one per line)"
            echo "  - \$INPUT_DIR/val.txt    (validation sequences, one per line)"
            echo ""
            echo "Default tokenizer: SentencePiece (BPE)"
            echo "  Train tokenizer: python scripts/train_tokenizer.py --vocab-size 128"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Determine tokenizer settings
if [ "$USE_NUCLEOTIDE_TOKENIZER" = true ]; then
    TOKENIZER_TYPE="dna-nucleotide"
    TOKENIZER_ARGS="--tokenizer-type dna-nucleotide"
    TOKENIZER_DISPLAY="DNANucleotideTokenizer (vocab_size=16)"
else
    TOKENIZER_TYPE="sentencepiece"
    TOKENIZER_ARGS="--tokenizer-type sentencepiece --tokenizer-model $TOKENIZER_MODEL"
    TOKENIZER_DISPLAY="SentencePieceTokenizer ($TOKENIZER_MODEL)"
fi

echo "=============================================="
echo "DNA Data Preparation for Megatron-LM"
echo "=============================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER_DISPLAY"
echo "=============================================="

# Check input files exist
if [ ! -f "$INPUT_DIR/train.txt" ]; then
    echo "Error: Training data not found at $INPUT_DIR/train.txt"
    exit 1
fi

# Check tokenizer model exists (for SentencePiece)
if [ "$USE_NUCLEOTIDE_TOKENIZER" = false ] && [ ! -f "$TOKENIZER_MODEL" ]; then
    echo "Error: SentencePiece tokenizer model not found at $TOKENIZER_MODEL"
    echo ""
    echo "Please train a tokenizer first:"
    echo "  python scripts/train_tokenizer.py \\"
    echo "    --input $INPUT_DIR/train.txt \\"
    echo "    --output-prefix tokenizer/dna_tokenizer \\"
    echo "    --vocab-size 128"
    echo ""
    echo "Or use the nucleotide tokenizer (no model needed):"
    echo "  $0 --nucleotide-tokenizer"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/megatron"

# Build verify flag
VERIFY_FLAG=""
if [ "$VERIFY" = true ]; then
    VERIFY_FLAG="--verify"
fi

# Process train split
echo ""
echo ">>> Processing train split..."
python "$PROJECT_DIR/scripts/preprocess_megatron.py" \
    --input "$INPUT_DIR/train.txt" \
    --output-prefix "$OUTPUT_DIR/megatron/train" \
    $TOKENIZER_ARGS \
    --log-interval 100000 \
    $VERIFY_FLAG

# Process validation split if it exists
if [ -f "$INPUT_DIR/val.txt" ]; then
    echo ""
    echo ">>> Processing validation split..."
    python "$PROJECT_DIR/scripts/preprocess_megatron.py" \
        --input "$INPUT_DIR/val.txt" \
        --output-prefix "$OUTPUT_DIR/megatron/validation" \
        $TOKENIZER_ARGS \
        --log-interval 100000 \
        $VERIFY_FLAG
fi

echo ""
echo "=============================================="
echo "Data Preparation Complete!"
echo "=============================================="
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR/megatron/"
echo ""
echo "To train with Megatron-LM, use:"
if [ "$USE_NUCLEOTIDE_TOKENIZER" = true ]; then
    echo "  ./training/pretrain_megatron.sh --data-path $OUTPUT_DIR/megatron/train_text_document --nucleotide-tokenizer"
else
    echo "  ./training/pretrain_megatron.sh --data-path $OUTPUT_DIR/megatron/train_text_document --tokenizer-model $TOKENIZER_MODEL"
fi
echo ""
echo "Tokenizer used: $TOKENIZER_DISPLAY"
