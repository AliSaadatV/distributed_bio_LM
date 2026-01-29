"""
Training script for DNA MoE Transformer models.

Supports distributed training with PyTorch DDP and integration with Megatron-LM.
Tracks GPU metrics including MFU, throughput, and memory utilization.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.moe_transformer import MoETransformer, MoETransformerConfig
from models.dna_tokenizer import DNATokenizer
from utils.gpu_metrics import GPUMetrics, compute_mfu, log_training_metrics


class DNADataset(Dataset):
    """Dataset for loading preprocessed DNA sequences."""
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 1024,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to preprocessed data directory or JSONL file
            seq_length: Sequence length
            split: Dataset split ('train', 'validation', 'test')
        """
        self.seq_length = seq_length
        self.split = split
        
        # Check if data_path is a directory or file
        if os.path.isdir(data_path):
            self._load_from_directory(data_path, split)
        else:
            self._load_from_jsonl(data_path)
    
    def _load_from_directory(self, data_dir: str, split: str):
        """Load from preprocessed binary files."""
        import numpy as np
        
        meta_file = os.path.join(data_dir, f"{split}_metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            
            bin_file = os.path.join(data_dir, f"{split}_text.bin")
            self.data = np.memmap(
                bin_file,
                dtype=np.uint16,
                mode="r"
            ).reshape(-1, self.seq_length)
            self.num_sequences = len(self.data)
        else:
            # Try JSONL format
            jsonl_file = os.path.join(data_dir, f"{split}.jsonl")
            self._load_from_jsonl(jsonl_file)
    
    def _load_from_jsonl(self, jsonl_path: str):
        """Load from JSONL file."""
        self.sequences = []
        self.tokenizer = DNATokenizer(model_max_length=self.seq_length)
        
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.sequences.append(item["text"])
        
        self.num_sequences = len(self.sequences)
        self.data = None
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.data is not None:
            # Binary format
            token_ids = torch.tensor(self.data[idx].copy(), dtype=torch.long)
        else:
            # JSONL format - tokenize on the fly
            sequence = self.sequences[idx]
            encoded = self.tokenizer(
                sequence,
                add_special_tokens=False,
                truncation=True,
                max_length=self.seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            token_ids = encoded["input_ids"].squeeze(0)
        
        return {
            "input_ids": token_ids,
            "labels": token_ids.clone(),
        }


def create_dataloader(
    data_path: str,
    batch_size: int,
    seq_length: int = 1024,
    split: str = "train",
    num_workers: int = 4,
    distributed: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        data_path: Path to data
        batch_size: Batch size per GPU
        seq_length: Sequence length
        split: Dataset split
        num_workers: Number of data loading workers
        distributed: Whether using distributed training
    
    Returns:
        DataLoader instance
    """
    dataset = DNADataset(data_path, seq_length, split)
    
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and not distributed),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def load_config(config_path: str) -> MoETransformerConfig:
    """Load model configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return MoETransformerConfig(**config_dict)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """Trainer class for DNA MoE models."""
    
    def __init__(
        self,
        model: MoETransformer,
        config: MoETransformerConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        args: argparse.Namespace,
        gpu_metrics: GPUMetrics,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.gpu_metrics = gpu_metrics
        self.rank = rank
        self.world_size = world_size
        
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.bf16 or args.fp16)
        
        # Logging
        if self.rank == 0:
            self.log_file = os.path.join(args.output_dir, "training_log.jsonl")
            os.makedirs(args.output_dir, exist_ok=True)
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        # Calculate total steps
        steps_per_epoch = len(self.train_dataloader)
        total_steps = self.args.max_steps or (steps_per_epoch * self.args.epochs)
        
        if self.rank == 0:
            print(f"\nStarting training:")
            print(f"  Total steps: {total_steps:,}")
            print(f"  Batch size per GPU: {self.args.batch_size}")
            print(f"  World size: {self.world_size}")
            print(f"  Global batch size: {self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}")
        
        accumulation_steps = self.args.gradient_accumulation_steps
        
        epoch = 0
        while self.global_step < total_steps:
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            for batch in self.train_dataloader:
                if self.global_step >= total_steps:
                    break
                
                # Move to device
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(
                    enabled=self.args.bf16 or self.args.fp16,
                    dtype=torch.bfloat16 if self.args.bf16 else torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"] / accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (self.global_step + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                # Update counters
                self.global_step += 1
                batch_tokens = input_ids.numel()
                self.tokens_seen += batch_tokens * self.world_size
                
                # Logging
                if self.global_step % self.args.log_interval == 0:
                    self._log_step(
                        loss=loss.item() * accumulation_steps,
                        aux_loss=outputs["aux_loss"],
                        z_loss=outputs["z_loss"],
                        batch_tokens=batch_tokens,
                    )
                
                # Validation
                if (
                    self.val_dataloader is not None
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self._validate()
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.args.save_interval == 0:
                    self._save_checkpoint()
            
            epoch += 1
        
        # Final save
        if self.rank == 0:
            self._save_checkpoint(final=True)
            print("\nTraining complete!")
    
    def _log_step(
        self,
        loss: float,
        aux_loss: float,
        z_loss: float,
        batch_tokens: int,
    ):
        """Log training metrics."""
        if self.rank != 0:
            return
        
        # Get timing and compute throughput
        self.gpu_metrics.step()
        elapsed = self.gpu_metrics.get_elapsed_time()
        tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
        samples_per_sec = self.global_step * self.args.batch_size * self.world_size / elapsed if elapsed > 0 else 0
        
        # Compute MFU
        mfu = compute_mfu(
            model_params=self.model.get_num_params(),
            batch_size=self.args.batch_size * self.world_size,
            seq_length=self.config.max_seq_length,
            elapsed_time=elapsed / self.global_step if self.global_step > 0 else 1,
            num_gpus=self.world_size,
        )
        
        # Get GPU memory
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        gpu_util = self.gpu_metrics.get_gpu_utilization()
        
        # Build log entry
        log_entry = {
            "step": self.global_step,
            "loss": loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "tokens_seen": self.tokens_seen,
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": samples_per_sec,
            "mfu": mfu,
            "gpu_memory_gb": gpu_memory,
            "gpu_util_percent": gpu_util,
            "elapsed_time": elapsed,
        }
        
        # Print
        print(
            f"Step {self.global_step:>6d} | "
            f"Loss: {loss:.4f} | "
            f"LR: {log_entry['lr']:.2e} | "
            f"Tokens/s: {tokens_per_sec:,.0f} | "
            f"MFU: {mfu:.1%} | "
            f"GPU Mem: {gpu_memory:.1f}GB"
        )
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _validate(self):
        """Run validation."""
        if self.val_dataloader is None:
            return
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                
                with torch.cuda.amp.autocast(
                    enabled=self.args.bf16 or self.args.fp16,
                    dtype=torch.bfloat16 if self.args.bf16 else torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
                
                total_loss += outputs["loss"].item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        
        if self.rank == 0:
            print(f"\nValidation loss: {avg_loss:.4f}")
            
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self._save_checkpoint(best=True)
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "config": self.config.__dict__,
            "args": vars(self.args),
        }
        
        if best:
            path = os.path.join(self.args.output_dir, "best_model.pt")
        elif final:
            path = os.path.join(self.args.output_dir, "final_model.pt")
        else:
            path = os.path.join(self.args.output_dir, f"checkpoint_{self.global_step}.pt")
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train DNA MoE Transformer")
    
    # Model arguments
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model-size", type=str, default=None, choices=["small", "medium"],
                        help="Use predefined model size instead of config")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation data")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max training tokens")
    
    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Precision arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    
    # Logging arguments
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save every N steps")
    
    # Other arguments
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("DNA MoE Transformer Training")
        print("=" * 60)
        print(f"Rank: {rank}, Local rank: {local_rank}, World size: {world_size}")
    
    # Load config
    if args.model_size:
        if args.model_size == "small":
            config = MoETransformerConfig.small_8m()
        else:
            config = MoETransformerConfig.medium_35m()
    else:
        config = load_config(args.config)
    
    if rank == 0:
        print(f"\nModel configuration:")
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
    
    # Create model
    model = MoETransformer(config).cuda()
    
    if rank == 0:
        total_params = model.get_num_params(non_embedding=False)
        active_params = model.get_num_active_params()
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Active: {active_params:,}")
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        args.data_path,
        args.batch_size,
        config.max_seq_length,
        "train",
        args.num_workers,
        distributed=(world_size > 1),
    )
    
    val_dataloader = None
    if args.val_data_path:
        val_dataloader = create_dataloader(
            args.val_data_path,
            args.batch_size,
            config.max_seq_length,
            "validation",
            args.num_workers,
            distributed=(world_size > 1),
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Create scheduler
    total_steps = args.max_steps or (len(train_dataloader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # GPU metrics
    gpu_metrics = GPUMetrics()
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        gpu_metrics=gpu_metrics,
        rank=rank,
        world_size=world_size,
    )
    
    try:
        trainer.train()
    finally:
        cleanup_distributed()
        
        if rank == 0:
            # Final metrics summary
            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            print(f"Total steps: {trainer.global_step:,}")
            print(f"Total tokens: {trainer.tokens_seen:,}")
            print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
