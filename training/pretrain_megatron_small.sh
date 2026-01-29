#!/bin/bash
# Train small DNA model (~8M params) with Megatron-LM
# Usage: ./training/pretrain_megatron_small.sh [--num-gpus N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/pretrain_megatron.sh" --config small "$@"
