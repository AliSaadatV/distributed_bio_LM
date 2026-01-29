"""
MoE Transformer model for DNA sequence modeling.

Architecture based on Genos but scaled down for smaller models:
- RMSNorm normalization
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation
- Mixture of Experts (MoE) FFN with top-k routing
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoETransformerConfig:
    """Configuration for MoE Transformer model."""
    
    # Model dimensions
    vocab_size: int = 16
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_query_groups: int = 2  # For GQA
    
    # MoE configuration
    num_experts: int = 4
    moe_top_k: int = 2
    moe_ffn_hidden_size: int = 512
    
    # Training configuration
    max_seq_length: int = 1024
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # RoPE configuration
    rope_base: int = 10000
    
    # MoE loss coefficients
    aux_loss_coeff: float = 1e-3
    z_loss_coeff: float = 1e-3
    
    # Precision
    use_flash_attention: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_query_groups == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_heads = self.num_query_groups
    
    @classmethod
    def small_8m(cls) -> "MoETransformerConfig":
        """Configuration for ~8M parameter model."""
        return cls(
            vocab_size=16,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            num_query_groups=2,
            num_experts=4,
            moe_top_k=2,
            moe_ffn_hidden_size=512,
            max_seq_length=1024,
        )
    
    @classmethod
    def medium_35m(cls) -> "MoETransformerConfig":
        """Configuration for ~35M parameter model."""
        return cls(
            vocab_size=16,
            hidden_size=512,
            num_layers=8,
            num_attention_heads=8,
            num_query_groups=4,
            num_experts=4,
            moe_top_k=2,
            moe_ffn_hidden_size=512,
            max_seq_length=1024,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_length: int = 1024, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin
        self._set_cos_sin_cache(max_seq_length)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_length:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Shares key-value heads across multiple query heads to reduce memory.
    """
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim = config.head_dim
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_length=config.max_seq_length,
            base=config.rope_base,
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for GQA.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] or [batch, 1, 1, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand K, V for GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.num_heads_per_kv, dim=1)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit."""
    
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: SiLU(W1(x)) * W3(x) -> W2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """Single expert in the MoE layer."""
    
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoERouter(nn.Module):
    """
    Router for Mixture of Experts.
    
    Implements top-k routing with auxiliary load balancing loss.
    """
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        
        # Router linear layer
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Loss tracking
        self.aux_loss = 0.0
        self.z_loss = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights.
        
        Args:
            hidden_states: [batch * seq_len, hidden_size]
        
        Returns:
            routing_weights: [batch * seq_len, top_k]
            selected_experts: [batch * seq_len, top_k]
            router_logits: [batch * seq_len, num_experts]
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch * seq_len, num_experts]
        
        # Compute routing weights (softmax over experts)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)
        
        # Compute auxiliary losses
        self._compute_aux_loss(routing_weights, router_logits)
        
        return topk_weights, topk_indices, router_logits
    
    def _compute_aux_loss(
        self,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        """Compute auxiliary load balancing and z-loss."""
        num_tokens = routing_weights.shape[0]
        
        # Load balancing loss (aux_loss)
        # Encourages uniform distribution of tokens across experts
        expert_mask = F.one_hot(
            torch.argmax(routing_weights, dim=-1),
            num_classes=self.num_experts
        ).float()
        tokens_per_expert = expert_mask.sum(dim=0)
        avg_tokens = num_tokens / self.num_experts
        
        # Fraction of tokens routed to each expert
        router_prob_per_expert = routing_weights.mean(dim=0)
        
        # Aux loss: encourages load balancing
        self.aux_loss = (
            self.num_experts 
            * (tokens_per_expert / num_tokens * router_prob_per_expert).sum()
        )
        
        # Z-loss: penalizes large router logits for stability
        self.z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Routes tokens to top-k experts and combines their outputs.
    """
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        
        # Router
        self.router = MoERouter(config)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_ffn_hidden_size)
            for _ in range(config.num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Auxiliary load balancing loss
            z_loss: Router z-loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_flat = hidden_states.view(-1, hidden_size)
        
        # Get routing decisions
        routing_weights, selected_experts, _ = self.router(hidden_flat)
        
        # Initialize output
        output = torch.zeros_like(hidden_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
            
            # Get token indices and their weights for this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Get weights for this expert
            expert_weights = torch.zeros(
                len(token_indices), device=hidden_states.device, dtype=hidden_states.dtype
            )
            for k in range(self.top_k):
                mask = selected_experts[token_indices, k] == expert_idx
                expert_weights[mask] = routing_weights[token_indices[mask], k]
            
            # Process through expert
            expert_input = hidden_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Add weighted output
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
        
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, self.router.aux_loss, self.router.z_loss


class TransformerBlock(nn.Module):
    """
    Single transformer block with MoE FFN.
    
    Architecture: RMSNorm -> Attention -> RMSNorm -> MoE FFN
    """
    
    def __init__(self, config: MoETransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.input_norm = RMSNorm(config.hidden_size)
        
        # Attention
        self.attention = GroupedQueryAttention(config)
        
        # Pre-FFN norm
        self.post_attention_norm = RMSNorm(config.hidden_size)
        
        # MoE FFN
        self.moe = MoELayer(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: MoE auxiliary loss
            z_loss: MoE z-loss
        """
        # Attention with residual
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states, aux_loss, z_loss = self.moe(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss, z_loss


class MoETransformer(nn.Module):
    """
    MoE Transformer model for DNA sequence modeling.
    
    Full model with embedding, transformer blocks, and LM head.
    """
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying (optional but common)
        # self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch, seq_len] - Token IDs
            attention_mask: [batch, seq_len] - Attention mask
            labels: [batch, seq_len] - Target token IDs for loss computation
        
        Returns:
            Dictionary with 'logits', 'loss', 'aux_loss', 'z_loss'
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Track MoE losses
        total_aux_loss = 0.0
        total_z_loss = 0.0
        
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states, aux_loss, z_loss = layer(hidden_states, attention_mask)
            total_aux_loss += aux_loss
            total_z_loss += z_loss
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add MoE losses
            avg_aux_loss = total_aux_loss / self.config.num_layers
            avg_z_loss = total_z_loss / self.config.num_layers
            
            loss = loss + self.config.aux_loss_coeff * avg_aux_loss
            loss = loss + self.config.z_loss_coeff * avg_z_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss / self.config.num_layers if total_aux_loss else 0.0,
            "z_loss": total_z_loss / self.config.num_layers if total_z_loss else 0.0,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: [batch, seq_len] - Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get predictions for the last token
            with torch.no_grad():
                outputs = self.forward(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if we hit max length
            if input_ids.size(1) >= self.config.max_seq_length:
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def get_num_active_params(self) -> int:
        """Get number of active parameters (with MoE sparsity)."""
        # Embedding and LM head
        active = self.embed_tokens.weight.numel()
        active += self.lm_head.weight.numel()
        active += self.norm.weight.numel()
        
        # Per layer
        for layer in self.layers:
            # Attention (always active)
            active += sum(p.numel() for p in layer.attention.parameters())
            active += layer.input_norm.weight.numel()
            active += layer.post_attention_norm.weight.numel()
            
            # Router
            active += layer.moe.router.gate.weight.numel()
            
            # Only top-k experts are active
            expert_params = sum(p.numel() for p in layer.moe.experts[0].parameters())
            active += expert_params * self.config.moe_top_k
        
        return active


def create_model(model_size: str = "small") -> MoETransformer:
    """
    Create a model with predefined configuration.
    
    Args:
        model_size: "small" (~8M params) or "medium" (~35M params)
    
    Returns:
        Configured MoETransformer model
    """
    if model_size == "small":
        config = MoETransformerConfig.small_8m()
    elif model_size == "medium":
        config = MoETransformerConfig.medium_35m()
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    return MoETransformer(config)


if __name__ == "__main__":
    # Test both model sizes
    for size in ["small", "medium"]:
        print(f"\n{'='*50}")
        print(f"Testing {size} model")
        print("="*50)
        
        model = create_model(size)
        config = model.config
        
        # Print model info
        total_params = model.get_num_params(non_embedding=False)
        active_params = model.get_num_active_params()
        
        print(f"Configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Query groups: {config.num_query_groups}")
        print(f"  Experts: {config.num_experts}")
        print(f"  Top-k: {config.moe_top_k}")
        print(f"  MoE FFN hidden: {config.moe_ffn_hidden_size}")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Active: {active_params:,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        print(f"\nForward pass:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Aux loss: {outputs['aux_loss']:.4f}")
        print(f"  Z loss: {outputs['z_loss']:.4f}")
