#!/bin/bash
# Train a SentencePiece BPE tokenizer for DNA sequences
#
# This script trains a BPE tokenizer with 128 vocabulary size (default),
# optimized for DNA sequences. The tokenizer learns common DNA patterns
# (dinucleotides, trinucleotides, common motifs).
#
# Usage:
#   ./scripts/train_dna_tokenizer.sh                          # Default: 128 vocab
#   ./scripts/train_dna_tokenizer.sh --vocab-size 256         # Custom vocab size
#   ./scripts/train_dna_tokenizer.sh --input my_data.txt      # Custom input file
#
# Output:
#   ./tokenizer/dna_tokenizer.model  - SentencePiece model file (use with --tokenizer-model)
#   ./tokenizer/dna_tokenizer.vocab  - Vocabulary file (human-readable)
#
# After training, preprocess data and train:
#   ./scripts/prepare_megatron_data.sh
#   ./training/pretrain_megatron.sh

set -e

# Default values
INPUT_FILE="./data/pretrain/train.txt"
OUTPUT_PREFIX="./tokenizer/dna_tokenizer"
VOCAB_SIZE=128
MODEL_TYPE="bpe"
SAMPLE_SIZE=0
LARGE_CORPUS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --large-corpus)
            LARGE_CORPUS="--large-corpus"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train a SentencePiece BPE tokenizer for DNA sequences."
            echo ""
            echo "Options:"
            echo "  --input FILE         Input file with DNA sequences (default: ./data/pretrain/train.txt)"
            echo "  --output-prefix PATH Output prefix (default: ./tokenizer/dna_tokenizer)"
            echo "  --vocab-size N       Vocabulary size (default: 128)"
            echo "  --model-type TYPE    Model type: bpe, unigram, char (default: bpe)"
            echo "  --sample-size N      Sample N sentences for training (default: 0 = all, use for large files)"
            echo "  --large-corpus       Enable memory-efficient mode for large datasets"
            echo ""
            echo "Output files:"
            echo "  {output-prefix}.model  - SentencePiece model"
            echo "  {output-prefix}.vocab  - Vocabulary file"
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

echo "=============================================="
echo "Training DNA SentencePiece Tokenizer"
echo "=============================================="
echo "Input: $INPUT_FILE"
echo "Output: ${OUTPUT_PREFIX}.model"
echo "Vocab size: $VOCAB_SIZE"
echo "Model type: $MODEL_TYPE"
if [ "$SAMPLE_SIZE" -gt 0 ]; then
    echo "Sample size: $SAMPLE_SIZE sentences"
fi
if [ -n "$LARGE_CORPUS" ]; then
    echo "Large corpus mode: enabled"
fi
echo "=============================================="

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    echo ""
    echo "Please provide a text file with DNA sequences (one per line)."
    echo "Example:"
    echo "  ATCGATCGATCGATCG..."
    echo "  GCTAGCTAGCTAGCTA..."
    exit 1
fi

# Build sample size argument
SAMPLE_ARG=""
if [ "$SAMPLE_SIZE" -gt 0 ]; then
    SAMPLE_ARG="--sample-size $SAMPLE_SIZE"
fi

# Train tokenizer
echo ""
echo ">>> Training tokenizer..."
python "$PROJECT_DIR/scripts/train_tokenizer.py" \
    --input "$INPUT_FILE" \
    --output-prefix "$OUTPUT_PREFIX" \
    --vocab-size "$VOCAB_SIZE" \
    --model-type "$MODEL_TYPE" \
    $SAMPLE_ARG \
    $LARGE_CORPUS

echo ""
echo "=============================================="
echo "Tokenizer Training Complete!"
echo "=============================================="
echo ""
echo "Files created:"
ls -lh "${OUTPUT_PREFIX}".* 2>/dev/null || echo "  (no files found)"
echo ""
echo "Next steps:"
echo "  1. Preprocess data with the tokenizer:"
echo "     ./scripts/prepare_megatron_data.sh --tokenizer-model ${OUTPUT_PREFIX}.model"
echo ""
echo "  2. Train the model:"
echo "     ./training/pretrain_megatron.sh --tokenizer-model ${OUTPUT_PREFIX}.model"
