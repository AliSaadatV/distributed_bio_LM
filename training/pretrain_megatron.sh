#!/bin/bash
# ============================================================================
# DNA Foundation Model Training with Megatron-LM
# ============================================================================
#
# This script trains a DNA MoE Transformer model using NVIDIA's Megatron-LM
# framework. It follows the same approach as the Genos project:
# https://github.com/BGI-HangzhouAI/Genos/blob/main/Scripts/Genos_pretrain_1B.sh
#
# Prerequisites:
#   1. Python 3.10+ (required by Megatron-LM)
#   2. Megatron-LM installed: pip install --no-build-isolation megatron-core[mlm,dev]
#   3. SentencePiece tokenizer trained (default): ./tokenizer/dna_tokenizer.model
#      Train with: python scripts/train_tokenizer.py --input data/pretrain/train.txt \
#                  --output-prefix tokenizer/dna_tokenizer --vocab-size 128
#   4. Data prepared with: python scripts/preprocess_megatron.py \
#          --input data/pretrain/train.txt --output-prefix processed_data/megatron/train
#
# Usage:
#   ./training/pretrain_megatron.sh [OPTIONS]
#
# Options:
#   --config small|medium|large   Model configuration (default: small)
#   --num-gpus N                  Number of GPUs (default: 4)
#   --data-path PATH              Path to preprocessed data (default: processed_data/megatron/train)
#   --checkpoint-dir PATH         Checkpoint save directory (default: checkpoints)
#   --tokenizer-model PATH        Path to SentencePiece .model file (default: ./tokenizer/dna_tokenizer.model)
#   --nucleotide-tokenizer        Use DNANucleotideTokenizer instead of SentencePiece
#   --wandb-project NAME          W&B project name (optional)
#   --dry-run                     Print command without executing
#   --debug                       Enable verbose NCCL debugging output
#   --test                        Quick test mode: 10 steps, small batch, no checkpoint
#
# Example:
#   ./training/pretrain_megatron.sh --config small --num-gpus 4
#   ./training/pretrain_megatron.sh --config medium --num-gpus 8 --wandb-project dna-foundation
#   ./training/pretrain_megatron.sh --config small --nucleotide-tokenizer  # Use nucleotide tokenizer
#
# Tokenizer (default: SentencePieceTokenizer):
#   Uses a trained BPE tokenizer (128 vocab by default) that learns DNA patterns.
#   Train with: python scripts/train_tokenizer.py --vocab-size 128
#
# Alternative Tokenizer (--nucleotide-tokenizer):
#   Uses DNANucleotideTokenizer with vocab_size=16:
#   - Special tokens: PAD(0), UNK(1), CLS(2), SEP(3), MASK(4), BOS(5), EOS(6)
#   - Nucleotides: A(7), T(8), C(9), G(10)
#   - Reserved: 11-15
#
# ============================================================================

set -e

# Default values
CONFIG="small"
NUM_GPUS=4
CHECKPOINT_DIR="./checkpoints"
DATA_PATH="./processed_data/megatron/train_text_document"
WANDB_PROJECT=""
DRY_RUN=false
DEBUG_NCCL=false
TEST_MODE=false

# Tokenizer settings (default: SentencePiece BPE tokenizer)
USE_NUCLEOTIDE_TOKENIZER=false
TOKENIZER_MODEL="./tokenizer/dna_tokenizer.model"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --debug)
            DEBUG_NCCL=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --tokenizer-model)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --nucleotide-tokenizer)
            USE_NUCLEOTIDE_TOKENIZER=true
            shift
            ;;
        -h|--help)
            head -55 "$0" | tail -50
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MEGATRON_DIR="$PROJECT_DIR/megatron_lm"

# Check prerequisites
if [ ! -d "$MEGATRON_DIR" ]; then
    echo "Error: Megatron-LM not found at $MEGATRON_DIR"
    echo "Please run: git clone https://github.com/NVIDIA/Megatron-LM.git megatron_lm"
    exit 1
fi

# Check tokenizer model exists (for SentencePiece)
if [ "$USE_NUCLEOTIDE_TOKENIZER" = false ] && [ ! -f "$TOKENIZER_MODEL" ]; then
    echo "Error: SentencePiece tokenizer model not found at $TOKENIZER_MODEL"
    echo ""
    echo "Please train a tokenizer first:"
    echo "  python scripts/train_tokenizer.py \\"
    echo "    --input data/pretrain/train.txt \\"
    echo "    --output-prefix tokenizer/dna_tokenizer \\"
    echo "    --vocab-size 128"
    echo ""
    echo "Or use the nucleotide tokenizer (no model needed):"
    echo "  ./training/pretrain_megatron.sh --nucleotide-tokenizer"
    exit 1
fi

# Check if data exists
if [ ! -f "${DATA_PATH}.bin" ]; then
    echo "Error: Data not found at ${DATA_PATH}.bin"
    echo ""
    echo "Please prepare data first using one of these methods:"
    echo ""
    echo "Method 1 (recommended - SentencePiece tokenizer):"
    echo "  python scripts/preprocess_megatron.py \\"
    echo "    --input data/pretrain/train.txt \\"
    echo "    --output-prefix processed_data/megatron/train \\"
    echo "    --tokenizer-type sentencepiece \\"
    echo "    --tokenizer-model ./tokenizer/dna_tokenizer.model"
    echo ""
    echo "Method 2 (nucleotide tokenizer):"
    echo "  python scripts/preprocess_megatron.py \\"
    echo "    --input data/pretrain/train.txt \\"
    echo "    --output-prefix processed_data/megatron/train \\"
    echo "    --tokenizer-type dna-nucleotide"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "Warning: Megatron-LM requires Python 3.10+. Current version: $PYTHON_VERSION"
    echo "Some features may not work correctly."
fi

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH}"

# NVIDIA CUDA library paths - required for transformer_engine to avoid slow recursive search
# transformer_engine searches CUDNN_HOME, NVRTC_HOME, CUDA_HOME etc. with recursive glob
# If any of these point to a deep directory tree (like /usr), it hangs forever
# Solution: Point library-specific *_HOME vars directly to the pip-installed nvidia packages
eval "$(python3 -c "
import os
import sys

# Map of transformer_engine lib names to nvidia pip package names
lib_map = {
    'cudnn': 'cudnn',
    'nvrtc': 'cuda_nvrtc',
    'curand': 'curand',
    'cublas': 'cublas',
    'cufft': 'cufft',
    'cusparse': 'cusparse',
    'cusolver': 'cusolver',
}

for lib_name, pkg_name in lib_map.items():
    try:
        mod = __import__(f'nvidia.{pkg_name}', fromlist=[pkg_name])
        lib_dir = os.path.dirname(mod.__file__) + '/lib'
        if os.path.isdir(lib_dir):
            # Set LIB_HOME to the lib directory so transformer_engine finds it immediately
            print(f'export {lib_name.upper()}_HOME=\"{lib_dir}\"')
    except:
        pass

# Build LD_LIBRARY_PATH with all nvidia libs
libs = []
for pkg in ['cudnn', 'cuda_runtime', 'cuda_nvrtc', 'cublas', 'cufft', 'nvjitlink', 'cusparse', 'curand', 'cusolver', 'nccl']:
    try:
        mod = __import__(f'nvidia.{pkg}', fromlist=[pkg])
        lib_path = os.path.dirname(mod.__file__) + '/lib'
        if os.path.isdir(lib_path):
            libs.append(lib_path)
    except:
        pass
if libs:
    print(f'export LD_LIBRARY_PATH=\"{\":\".join(libs)}:\$LD_LIBRARY_PATH\"')
" 2>/dev/null)"

# NCCL configuration for multi-GPU training
# Use shared memory for single-node communication (faster and more reliable)
if [ "$DEBUG_NCCL" = true ]; then
    export NCCL_DEBUG=INFO  # Verbose debugging output
    export NCCL_DEBUG_SUBSYS=ALL
else
    export NCCL_DEBUG=WARN  # Only warnings
fi
export NCCL_TIMEOUT=600  # 10 minute timeout (default is 30 min which is too long)
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_LEVEL=NVL  # Use NVLink if available, fallback to PCIe

# Prevent OMP thread oversubscription warning
export OMP_NUM_THREADS=1

# Set master address for distributed training (localhost for single node)
# export MASTER_ADDR=localhost
# export MASTER_PORT=${MASTER_PORT:-29500}

# Common training parameters
# Using 512 to match input data (512bp sequences)
SEQ_LENGTH=512
LR=1e-4
MIN_LR=1e-5
WEIGHT_DECAY=0.1
WARMUP_FRACTION=0.1
CLIP_GRAD=1.0

# Configuration-specific parameters
# Batch sizes optimized for 512 sequence length and respective model sizes
case $CONFIG in
    small)
        # Small model (~8M params)
        # Larger micro-batch possible due to small model size
        NUM_LAYERS=4
        HIDDEN_SIZE=256
        FFN_HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=4
        NUM_QUERY_GROUPS=2
        NUM_EXPERTS=4
        MOE_TOP_K=2
        MICRO_BATCH_SIZE=64
        GLOBAL_BATCH_SIZE=512
        TRAIN_SAMPLES=10000000
        ;;
    medium)
        # Medium model (~35M params)
        # Balanced micro-batch for memory and throughput
        NUM_LAYERS=8
        HIDDEN_SIZE=512
        FFN_HIDDEN_SIZE=2048
        NUM_ATTENTION_HEADS=8
        NUM_QUERY_GROUPS=4
        NUM_EXPERTS=4
        MOE_TOP_K=2
        MICRO_BATCH_SIZE=32
        GLOBAL_BATCH_SIZE=512
        TRAIN_SAMPLES=100000000
        ;;
    large)
        # Large model (~1.2B params, similar to Genos)
        # Matches Genos settings for 1B+ scale
        NUM_LAYERS=12
        HIDDEN_SIZE=1024
        FFN_HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=16
        NUM_QUERY_GROUPS=8
        NUM_EXPERTS=8
        MOE_TOP_K=2
        MICRO_BATCH_SIZE=1
        GLOBAL_BATCH_SIZE=1024
        TRAIN_SAMPLES=1000000000
        ;;
    *)
        echo "Error: Unknown config '$CONFIG'. Use small, medium, or large."
        exit 1
        ;;
esac

# Test mode overrides - quick validation run
if [ "$TEST_MODE" = true ]; then
    TRAIN_SAMPLES=100       # Just enough for 10 steps
    MICRO_BATCH_SIZE=2      # Small batch
    GLOBAL_BATCH_SIZE=8     # Small global batch
    echo "TEST MODE: Running quick validation (10 steps)"
fi

LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 80 / 100))

# Build argument groups (following Genos patterns)
MODEL_ARGS=" \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --no-masked-softmax-fusion \
    --no-position-embedding"

MOE_ARGS=" \
    --num-experts $NUM_EXPERTS \
    --moe-router-topk $MOE_TOP_K \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-router-dtype fp32 \
    --moe-z-loss-coeff 1e-3"
# Note: Removed --moe-grouped-gemm to avoid potential CUDA kernel issues
# Re-enable for better performance after confirming training works

# Data arguments - tokenizer selection based on flag
# Note: --num-workers 8 for performance (Genos uses 8)
if [ "$USE_NUCLEOTIDE_TOKENIZER" = true ]; then
    # Use single-nucleotide tokenizer (vocab_size=16, no model file needed)
    TOKENIZER_TYPE="DNANucleotideTokenizer"
    TOKENIZER_ARGS="--tokenizer-type ${TOKENIZER_TYPE}"
    TOKENIZER_DISPLAY="DNANucleotideTokenizer (vocab_size=16)"
else
    # Use SentencePiece BPE tokenizer (default, matches Genos)
    TOKENIZER_TYPE="SentencePieceTokenizer"
    TOKENIZER_ARGS="--tokenizer-type ${TOKENIZER_TYPE} --tokenizer-model ${TOKENIZER_MODEL}"
    TOKENIZER_DISPLAY="SentencePieceTokenizer (${TOKENIZER_MODEL})"
fi

DATA_ARGS=" \
    --num-workers 4 \
    --dataloader-type cyclic \
    ${TOKENIZER_ARGS} \
    --data-path ${DATA_PATH} \
    --split 980,20,0 \
    --no-create-attention-mask-in-dataloader"

TRAINING_ARGS=" \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr ${LR} \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-decay-style cosine \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --lr-warmup-fraction ${WARMUP_FRACTION} \
    --clip-grad ${CLIP_GRAD} \
    --bf16 \
    --use-flash-attn \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --disable-bf16-reduced-precision-matmul \
    --no-gradient-accumulation-fusion"

# Single-node parallelism (for 1-8 GPUs)
MODEL_PARALLEL_ARGS=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --use-distributed-optimizer"

# Logging args - reduced in test mode
if [ "$TEST_MODE" = true ]; then
    LOGGING_ARGS=" \
        --log-interval 1 \
        --save-interval 99999 \
        --eval-interval 99999 \
        --eval-iters 0 \
        --no-save-optim \
        --no-save-rng \
        --log-throughput"
else
    LOGGING_ARGS=" \
        --moe-per-layer-logging \
        --log-interval 10 \
        --save-interval 1000 \
        --eval-interval 500 \
        --eval-iters 100 \
        --save $CHECKPOINT_DIR \
        --tensorboard-dir ${CHECKPOINT_DIR}/tensorboard \
        --log-throughput"
    
    # Add W&B if specified
    if [ -n "$WANDB_PROJECT" ]; then
        LOGGING_ARGS="$LOGGING_ARGS --wandb-project $WANDB_PROJECT --wandb-exp-name dna-${CONFIG}"
    fi
fi

# Print configuration
echo "=============================================="
echo "DNA Foundation Model Training with Megatron-LM"
echo "=============================================="
echo "Configuration: $CONFIG"
echo "  Layers: $NUM_LAYERS"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  FFN hidden size: $FFN_HIDDEN_SIZE"
echo "  Attention heads: $NUM_ATTENTION_HEADS"
echo "  Query groups (GQA): $NUM_QUERY_GROUPS"
echo "  Experts: $NUM_EXPERTS"
echo "  Top-k: $MOE_TOP_K"
echo ""
echo "Training:"
echo "  GPUs: $NUM_GPUS"
echo "  Micro batch: $MICRO_BATCH_SIZE"
echo "  Global batch: $GLOBAL_BATCH_SIZE"
echo "  Sequence length: $SEQ_LENGTH"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Flash Attention: Enabled"
echo ""
echo "Data:"
echo "  Data path: $DATA_PATH"
echo "  Tokenizer: $TOKENIZER_DISPLAY"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "=============================================="

# Build full command
# Use --standalone for single-node training (more reliable than default rendezvous)
# --rdzv_backend=c10d is the modern PyTorch rendezvous backend
# CMD="torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:${MASTER_PORT} \
#     ${MEGATRON_DIR}/pretrain_gpt.py \
#     ${MODEL_ARGS} \
#     ${MOE_ARGS} \
#     ${DATA_ARGS} \
#     ${TRAINING_ARGS} \
#     ${MODEL_PARALLEL_ARGS} \
#     ${LOGGING_ARGS}"

CMD="torchrun \
  --standalone \
  --nproc_per_node=${NUM_GPUS} \
  --log-dir ${CHECKPOINT_DIR}/torchrun_logs \
  --tee 3 \
  --max-restarts 0 \
  ${MEGATRON_DIR}/pretrain_gpt.py \
  ${MODEL_ARGS} \
  ${MOE_ARGS} \
  ${DATA_ARGS} \
  ${TRAINING_ARGS} \
  ${MODEL_PARALLEL_ARGS} \
  ${LOGGING_ARGS}"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Command that would be executed:"
    echo "=============================================="
    echo "$CMD"
    echo "=============================================="
else
    echo ""
    echo "Starting training..."
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
    echo ""
    # PyTorch distributed logging (DETAIL is very verbose, use only for debugging)
    if [ "$DEBUG_NCCL" = true ]; then
        export TORCH_CPP_LOG_LEVEL=INFO
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
    else
        export TORCH_CPP_LOG_LEVEL=WARNING
        export TORCH_DISTRIBUTED_DEBUG=OFF
    fi
    eval "$CMD"
fi
