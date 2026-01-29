#!/bin/bash
# Train medium DNA model (~35M params) with Megatron-LM
# Usage: ./training/pretrain_megatron_medium.sh [--num-gpus N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/pretrain_megatron.sh" --config medium "$@"
