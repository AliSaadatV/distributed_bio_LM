# DNA Foundation Models

Small-scale DNA language models based on the Genos architecture, designed for genomic sequence modeling using Mixture of Experts (MoE) transformers.

## Model Specifications

### Small Model (~8M Parameters)

| Attribute | Value |
|-----------|-------|
| Total Parameters | ~8M |
| Active Parameters | ~4M |
| Training Tokens | 160M |
| Architecture | MoE Transformer |
| Number of Experts | 4 |
| Top-k | 2 |
| Layers | 4 |
| Attention Hidden | 256 |
| Attention Heads | 4 |
| Query Groups (GQA) | 2 |
| MoE FFN Hidden | 512 |
| Vocabulary | 128 (SentencePiece BPE) |
| Max Context Length | 512 |

### Medium Model (~35M Parameters)

| Attribute | Value |
|-----------|-------|
| Total Parameters | ~35M |
| Active Parameters | ~18M |
| Training Tokens | 700M |
| Architecture | MoE Transformer |
| Number of Experts | 4 |
| Top-k | 2 |
| Layers | 8 |
| Attention Hidden | 512 |
| Attention Heads | 8 |
| Query Groups (GQA) | 4 |
| MoE FFN Hidden | 512 |
| Vocabulary | 128 (SentencePiece BPE) |
| Max Context Length | 512 |

## Architecture

Both models use the same architectural components:

- **Tokenization**: SentencePiece BPE (learns DNA patterns from data)
- **Vocabulary**: 128 tokens (default) or 16 tokens (nucleotide tokenizer)
- **Normalization**: RMSNorm (pre-normalization)
- **Position Encoding**: Rotary Position Embedding (RoPE)
- **Attention**: Grouped Query Attention (GQA) for efficient KV caching
- **Activation**: SwiGLU in FFN layers
- **Expert Routing**: Top-2 with auxiliary load balancing loss

### Tokenizer Options

| Tokenizer | Vocab Size | Description |
|-----------|------------|-------------|
| SentencePiece BPE | 128 | Default. Learns subword patterns from DNA sequences |
| DNANucleotideTokenizer | 16 | Character-level (A=7, T=8, C=9, G=10) + special tokens |

The SentencePiece tokenizer is recommended as it can learn meaningful DNA patterns (motifs, repeats) and achieves better compression.

## Prerequisites

- **Python**: 3.10+ (required by Megatron-LM)
- **CUDA**: 11.8+ with cuDNN
- **PyTorch**: 2.1.0+
- **GPUs**: NVIDIA GPUs with sufficient memory (A100/H100 recommended for larger models)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Genos

# Install dependencies
pip install -r requirements.txt

# Install Megatron-LM (from included submodule)
cd megatron_lm && pip install --no-build-isolation .[mlm,dev] && cd ..

# Optional: Install Flash Attention for better performance
pip install flash-attn --no-build-isolation
```

### Quick Setup Verification

```bash
# Verify Megatron-LM installation
python -c "from megatron.core import mpu; print('Megatron-LM OK')"

# Test training script (dry run)
./training/pretrain_megatron.sh --config small --num-gpus 1 --dry-run
```

## Data Preparation

The models are trained on multi-species genomes from the DNABERT-2 pretraining dataset ([InstaDeepAI/multi_species_genomes](https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes)), following the approach described in the DNABERT-2 paper (ICLR 2024).

### Data Format

Training data should be placed in `data/pretrain/` as plain text files:
- `train.txt` - Training sequences (one 512bp DNA sequence per line)
- `val.txt` - Validation sequences

Each line should contain only canonical nucleotides (A, C, T, G).

### Preprocessing for Megatron-LM

```bash
# Step 1: Train the SentencePiece tokenizer (BPE, vocab_size=128)
python scripts/train_tokenizer.py \
    --input data/pretrain/train.txt \
    --output-prefix tokenizer/dna_tokenizer \
    --vocab-size 128

# Step 2: Preprocess data to Megatron binary format
python scripts/preprocess_megatron.py \
    --input data/pretrain/train.txt \
    --output-prefix processed_data/megatron/train \
    --tokenizer-type sentencepiece \
    --tokenizer-model ./tokenizer/dna_tokenizer.model

# Step 3: Preprocess validation data
python scripts/preprocess_megatron.py \
    --input data/pretrain/val.txt \
    --output-prefix processed_data/megatron/val \
    --tokenizer-type sentencepiece \
    --tokenizer-model ./tokenizer/dna_tokenizer.model
```

This creates Megatron-compatible binary files (`.bin` and `.idx`) for efficient data loading during training.

## Training

Training uses NVIDIA's Megatron-LM framework for efficient distributed training with MoE support.

### Quick Start with Megatron-LM

```bash
# Train small model (~8M params) on 4 GPUs
./training/pretrain_megatron.sh --config small --num-gpus 4

# Train medium model (~35M params) on 8 GPUs
./training/pretrain_megatron.sh --config medium --num-gpus 8

# Train with W&B logging
./training/pretrain_megatron.sh --config small --num-gpus 4 --wandb-project dna-foundation
```

### Training Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | small | Model config: `small`, `medium`, or `large` |
| `--num-gpus` | 4 | Number of GPUs to use |
| `--data-path` | processed_data/megatron/train | Path to preprocessed data |
| `--checkpoint-dir` | checkpoints | Directory for saving checkpoints |
| `--tokenizer-model` | ./tokenizer/dna_tokenizer.model | Path to SentencePiece model |
| `--nucleotide-tokenizer` | - | Use DNANucleotideTokenizer instead of SentencePiece |
| `--wandb-project` | - | W&B project name for logging |
| `--dry-run` | - | Print command without executing |
| `--test` | - | Quick test mode (10 steps) |

### Model Configurations

| Config | Layers | Hidden | FFN | Heads | Experts | Micro Batch | Global Batch |
|--------|--------|--------|-----|-------|---------|-------------|--------------|
| small | 4 | 256 | 1024 | 4 | 4 | 64 | 512 |
| medium | 8 | 512 | 2048 | 8 | 4 | 32 | 512 |
| large | 12 | 1024 | 4096 | 16 | 8 | 1 | 1024 |

### Training Features

- **Distributed Training**: Multi-GPU training via `torchrun`
- **Mixed Precision**: bfloat16 with Flash Attention
- **MoE Support**: Top-k routing with auxiliary load balancing loss
- **Parallelism**: Tensor, Pipeline, Expert, and Data parallelism options
- **Optimizer**: Distributed AdamW with cosine LR decay

### Alternative: Custom PyTorch Training

For simpler setups without Megatron-LM:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py \
    --config configs/small_8m.yaml \
    --data-path ./data/processed_data \
    --output-dir ./output/custom \
    --batch-size 32 \
    --lr 1e-4 \
    --bf16
```

## Project Structure

```
Genos/
├── configs/
│   ├── small_8m.yaml              # Small model configuration
│   └── medium_35m.yaml            # Medium model configuration
├── data/
│   ├── prepare_data.py            # Data preparation script
│   └── pretrain/
│       ├── train.txt              # Training sequences
│       └── val.txt                # Validation sequences
├── models/
│   ├── dna_tokenizer.py           # DNA tokenizer (HuggingFace compatible)
│   └── moe_transformer.py         # MoE Transformer implementation
├── scripts/
│   ├── preprocess_megatron.py     # Data preprocessing for Megatron
│   └── train_tokenizer.py         # SentencePiece tokenizer training
├── training/
│   ├── train.py                   # Custom PyTorch training script
│   └── pretrain_megatron.sh       # Megatron-LM training script
├── megatron_lm/                   # NVIDIA Megatron-LM framework
│   ├── pretrain_gpt.py            # Main Megatron pretraining entry
│   └── megatron/                  # Core Megatron modules
├── utils/
│   └── gpu_metrics.py             # GPU monitoring and MFU calculation
├── processed_data/                # Preprocessed training data
├── tokenizer/                     # Tokenizer files directory
│   ├── dna_tokenizer.model        # Trained SentencePiece model
│   └── dna_tokenizer.vocab        # Vocabulary file
├── requirements.txt
└── README.md
```

## GPU Metrics

The training script tracks several GPU metrics:

- **MFU (Model FLOPS Utilization)**: Ratio of achieved FLOPS to theoretical peak
- **Throughput**: Tokens/second and samples/second
- **GPU Memory**: Peak and current allocation
- **GPU Utilization**: Percentage via nvidia-smi

Metrics are logged to `training_log.jsonl` in the output directory.


## Key architectural choices:

- Mixture of Experts from Genos and Switch Transformer
- Grouped Query Attention from Llama 2
- RoPE from RoFormer
- SwiGLU from PaLM


