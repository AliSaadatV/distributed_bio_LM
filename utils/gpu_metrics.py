"""
GPU Metrics tracking for DNA model training.

Provides utilities for measuring:
- MFU (Model FLOPS Utilization)
- Throughput (tokens/second, samples/second)
- GPU memory utilization
- GPU compute utilization
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# H100 specifications
H100_PEAK_FLOPS_FP16 = 1979e12  # 1979 TFLOPS for FP16/BF16
H100_PEAK_FLOPS_FP32 = 989e12   # 989 TFLOPS for FP32
H100_MEMORY_GB = 80  # 80GB HBM3 (or 94GB for H100 NVL)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    loss: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    mfu: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    elapsed_time: float = 0.0


class GPUMetrics:
    """
    GPU metrics tracker for training.
    
    Tracks:
    - Timing for throughput calculation
    - GPU memory usage
    - GPU utilization via nvidia-smi
    """
    
    def __init__(self, gpu_memory_gb: float = H100_MEMORY_GB):
        """
        Initialize GPU metrics tracker.
        
        Args:
            gpu_memory_gb: Total GPU memory in GB
        """
        self.gpu_memory_gb = gpu_memory_gb
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.last_step_time = time.time()
        
        # Initialize NVML for GPU utilization tracking
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) 
                    for i in range(self.device_count)
                ]
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
        
        # History for averaging
        self.throughput_history: List[float] = []
        self.mfu_history: List[float] = []
        self.memory_history: List[float] = []
    
    def step(self):
        """Mark the end of a training step."""
        current_time = time.time()
        self.last_step_time = current_time
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since start."""
        return time.time() - self.start_time
    
    def get_step_time(self) -> float:
        """Get time since last step."""
        return time.time() - self.step_start_time
    
    def reset_step_timer(self):
        """Reset the step timer."""
        self.step_start_time = time.time()
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        Get GPU memory information.
        
        Returns:
            Dictionary with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def get_gpu_utilization(self) -> float:
        """
        Get GPU utilization percentage.
        
        Returns:
            Average GPU utilization across all GPUs (0-100)
        """
        if not self.nvml_initialized:
            return 0.0
        
        try:
            total_util = 0.0
            for handle in self.handles:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total_util += util.gpu
            return total_util / len(self.handles)
        except Exception:
            return 0.0
    
    def get_gpu_power(self) -> float:
        """
        Get total GPU power consumption in watts.
        
        Returns:
            Total power consumption across all GPUs
        """
        if not self.nvml_initialized:
            return 0.0
        
        try:
            total_power = 0.0
            for handle in self.handles:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                total_power += power
            return total_power
        except Exception:
            return 0.0
    
    def get_gpu_temperature(self) -> List[float]:
        """
        Get GPU temperatures.
        
        Returns:
            List of temperatures for each GPU
        """
        if not self.nvml_initialized:
            return []
        
        try:
            temps = []
            for handle in self.handles:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                temps.append(temp)
            return temps
        except Exception:
            return []
    
    def cleanup(self):
        """Clean up NVML resources."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def compute_model_flops(
    num_params: int,
    seq_length: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    num_experts: int = 1,
    moe_top_k: int = 1,
    ffn_hidden_size: Optional[int] = None,
) -> int:
    """
    Estimate FLOPS per forward pass for a transformer model.
    
    This is a rough estimate based on the dominant operations:
    - Attention: 4 * batch * seq^2 * hidden (per layer)
    - FFN: 2 * batch * seq * hidden * ffn_hidden * 2 (per layer, with SwiGLU)
    - Embedding: batch * seq * hidden * vocab
    
    Args:
        num_params: Total model parameters
        seq_length: Sequence length
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        num_experts: Number of MoE experts
        moe_top_k: Number of active experts per token
        ffn_hidden_size: FFN hidden size (default: 4 * hidden_size)
    
    Returns:
        Estimated FLOPS per forward pass
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size
    
    # Embedding FLOPS (lookup + output projection)
    embed_flops = 2 * seq_length * hidden_size * vocab_size
    
    # Per-layer FLOPS
    # Attention: QKV projection + attention scores + output projection
    attn_flops = (
        # QKV projection
        3 * seq_length * hidden_size * hidden_size +
        # Attention scores (Q @ K^T)
        2 * seq_length * seq_length * hidden_size +
        # Attention @ V
        2 * seq_length * seq_length * hidden_size +
        # Output projection
        seq_length * hidden_size * hidden_size
    )
    
    # FFN FLOPS (SwiGLU has 3 projections)
    # With MoE, only top_k experts are active
    ffn_flops = 3 * 2 * seq_length * hidden_size * ffn_hidden_size * moe_top_k
    
    # Total per layer
    layer_flops = attn_flops + ffn_flops
    
    # Total forward pass
    forward_flops = embed_flops + num_layers * layer_flops
    
    # Training includes forward + backward (roughly 3x forward)
    training_flops = 3 * forward_flops
    
    return training_flops


def compute_mfu(
    model_params: int,
    batch_size: int,
    seq_length: int,
    elapsed_time: float,
    num_gpus: int = 1,
    peak_flops: float = H100_PEAK_FLOPS_FP16,
    num_layers: int = 4,
    hidden_size: int = 256,
    num_attention_heads: int = 4,
    vocab_size: int = 16,
    num_experts: int = 4,
    moe_top_k: int = 2,
    ffn_hidden_size: int = 512,
) -> float:
    """
    Compute Model FLOPS Utilization (MFU).
    
    MFU = (achieved FLOPS) / (theoretical peak FLOPS)
    
    Args:
        model_params: Total model parameters
        batch_size: Global batch size
        seq_length: Sequence length
        elapsed_time: Time per step in seconds
        num_gpus: Number of GPUs
        peak_flops: Theoretical peak FLOPS per GPU
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        num_experts: Number of MoE experts
        moe_top_k: Number of active experts
        ffn_hidden_size: FFN hidden size
    
    Returns:
        MFU as a fraction (0-1)
    """
    if elapsed_time <= 0:
        return 0.0
    
    # Compute FLOPS per sample
    flops_per_sample = compute_model_flops(
        model_params,
        seq_length,
        num_layers,
        hidden_size,
        num_attention_heads,
        vocab_size,
        num_experts,
        moe_top_k,
        ffn_hidden_size,
    )
    
    # Total FLOPS for the batch
    total_flops = flops_per_sample * batch_size
    
    # Achieved FLOPS
    achieved_flops = total_flops / elapsed_time
    
    # Theoretical peak across all GPUs
    total_peak_flops = peak_flops * num_gpus
    
    # MFU
    mfu = achieved_flops / total_peak_flops
    
    return min(mfu, 1.0)  # Cap at 1.0


def log_training_metrics(
    step: int,
    loss: float,
    tokens_seen: int,
    elapsed_time: float,
    model_params: int,
    batch_size: int,
    seq_length: int,
    num_gpus: int,
    gpu_metrics: Optional[GPUMetrics] = None,
    config: Optional[dict] = None,
) -> TrainingMetrics:
    """
    Compute and log comprehensive training metrics.
    
    Args:
        step: Current training step
        loss: Current loss value
        tokens_seen: Total tokens processed
        elapsed_time: Total elapsed time
        model_params: Model parameters
        batch_size: Global batch size
        seq_length: Sequence length
        num_gpus: Number of GPUs
        gpu_metrics: Optional GPUMetrics instance
        config: Optional model config dict
    
    Returns:
        TrainingMetrics dataclass with all metrics
    """
    # Throughput
    tokens_per_second = tokens_seen / elapsed_time if elapsed_time > 0 else 0
    samples_per_second = (step * batch_size) / elapsed_time if elapsed_time > 0 else 0
    
    # MFU
    time_per_step = elapsed_time / step if step > 0 else 1.0
    
    # Get config values or use defaults
    num_layers = config.get("num_layers", 4) if config else 4
    hidden_size = config.get("hidden_size", 256) if config else 256
    num_attention_heads = config.get("num_attention_heads", 4) if config else 4
    vocab_size = config.get("vocab_size", 16) if config else 16
    num_experts = config.get("num_experts", 4) if config else 4
    moe_top_k = config.get("moe_top_k", 2) if config else 2
    ffn_hidden_size = config.get("moe_ffn_hidden_size", 512) if config else 512
    
    mfu = compute_mfu(
        model_params,
        batch_size,
        seq_length,
        time_per_step,
        num_gpus,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        num_experts=num_experts,
        moe_top_k=moe_top_k,
        ffn_hidden_size=ffn_hidden_size,
    )
    
    # GPU memory
    gpu_memory_allocated = 0.0
    gpu_memory_used = 0.0
    gpu_utilization = 0.0
    
    if gpu_metrics:
        mem_info = gpu_metrics.get_gpu_memory_info()
        gpu_memory_allocated = mem_info.get("allocated", 0.0)
        gpu_memory_used = mem_info.get("max_allocated", 0.0)
        gpu_utilization = gpu_metrics.get_gpu_utilization()
    elif torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_memory_used = torch.cuda.max_memory_allocated() / 1e9
    
    return TrainingMetrics(
        step=step,
        loss=loss,
        tokens_per_second=tokens_per_second,
        samples_per_second=samples_per_second,
        mfu=mfu,
        gpu_memory_used_gb=gpu_memory_used,
        gpu_memory_allocated_gb=gpu_memory_allocated,
        gpu_utilization_percent=gpu_utilization,
        elapsed_time=elapsed_time,
    )


def print_metrics_summary(metrics: TrainingMetrics):
    """Print a formatted summary of training metrics."""
    print("\n" + "=" * 60)
    print("Training Metrics Summary")
    print("=" * 60)
    print(f"Step: {metrics.step:,}")
    print(f"Loss: {metrics.loss:.4f}")
    print(f"Throughput:")
    print(f"  - Tokens/sec: {metrics.tokens_per_second:,.0f}")
    print(f"  - Samples/sec: {metrics.samples_per_second:,.2f}")
    print(f"MFU: {metrics.mfu:.1%}")
    print(f"GPU Memory:")
    print(f"  - Used: {metrics.gpu_memory_used_gb:.2f} GB")
    print(f"  - Allocated: {metrics.gpu_memory_allocated_gb:.2f} GB")
    print(f"GPU Utilization: {metrics.gpu_utilization_percent:.1f}%")
    print(f"Elapsed Time: {metrics.elapsed_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    # Test GPU metrics
    print("Testing GPU Metrics")
    print("=" * 60)
    
    gpu_metrics = GPUMetrics()
    
    # Memory info
    mem_info = gpu_metrics.get_gpu_memory_info()
    print(f"GPU Memory: {mem_info}")
    
    # Utilization
    util = gpu_metrics.get_gpu_utilization()
    print(f"GPU Utilization: {util:.1f}%")
    
    # Power
    power = gpu_metrics.get_gpu_power()
    print(f"GPU Power: {power:.1f} W")
    
    # Temperature
    temps = gpu_metrics.get_gpu_temperature()
    print(f"GPU Temperatures: {temps}")
    
    # Test MFU calculation
    print("\n" + "=" * 60)
    print("Testing MFU Calculation")
    print("=" * 60)
    
    # Small model
    mfu_small = compute_mfu(
        model_params=8_000_000,
        batch_size=512,
        seq_length=1024,
        elapsed_time=1.0,  # 1 second per step
        num_gpus=4,
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        vocab_size=16,
        num_experts=4,
        moe_top_k=2,
        ffn_hidden_size=512,
    )
    print(f"Small model MFU (4 H100s): {mfu_small:.1%}")
    
    # Medium model
    mfu_medium = compute_mfu(
        model_params=35_000_000,
        batch_size=256,
        seq_length=1024,
        elapsed_time=1.0,
        num_gpus=4,
        num_layers=8,
        hidden_size=512,
        num_attention_heads=8,
        vocab_size=16,
        num_experts=4,
        moe_top_k=2,
        ffn_hidden_size=512,
    )
    print(f"Medium model MFU (4 H100s): {mfu_medium:.1%}")
    
    gpu_metrics.cleanup()
