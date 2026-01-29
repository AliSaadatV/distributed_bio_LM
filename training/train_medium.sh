#!/bin/bash
# Training script for Medium DNA Model (~35M parameters)
# Uses 4 H100 GPUs with DDP
# Target: 700M training tokens (20x parameters)

set -e

# Configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

# Paths (adjust these for your setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/medium_35m.yaml"
DATA_PATH="${1:-${PROJECT_DIR}/data/processed_data}"
OUTPUT_DIR="${2:-${PROJECT_DIR}/output/medium_35m}"

# Training hyperparameters
# Medium model: ~35M params, ~18M active
# Training tokens: 700M (20x params)
# Sequence length: 1024
# Tokens per sample: 1024
# Samples needed: 700M / 1024 ≈ 684K samples
# With batch size 16 * 4 GPUs * 4 accum = 256 global batch
# Steps needed: 684K / 256 ≈ 2672 steps

BATCH_SIZE=16  # Smaller batch size for larger model
GRADIENT_ACCUMULATION=4
MAX_STEPS=2672  # ~700M tokens with global batch 256
LR=1e-4
MIN_LR=1e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log configuration
echo "=========================================="
echo "Medium DNA Model Training (35M params)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Max steps: $MAX_STEPS"
echo "Learning rate: $LR"
echo "=========================================="

# Check for data
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data directory not found at $DATA_PATH"
    echo "Please run: python data/prepare_data.py --output-dir $DATA_PATH"
    exit 1
fi

# Run training with 4 GPUs
torchrun \
    --standalone \
    --nproc_per_node=4 \
    "${SCRIPT_DIR}/train.py" \
    --config "$CONFIG_FILE" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --max-steps $MAX_STEPS \
    --lr $LR \
    --min-lr $MIN_LR \
    --warmup-ratio $WARMUP_RATIO \
    --weight-decay $WEIGHT_DECAY \
    --grad-clip $GRAD_CLIP \
    --bf16 \
    --log-interval 10 \
    --eval-interval 200 \
    --save-interval 500 \
    --num-workers 4 \
    --seed 42

echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
