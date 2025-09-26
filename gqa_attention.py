import torch
import torch.nn as nn

from config import GPTConfig
from rotary_embeddings import Rotary, apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, n_kv_head: int) -> None:
        super().__init__()

        # Validate configuration
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        assert n_head % n_kv_head == 0, (
            f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        )

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.num_query_groups = n_head // n_kv_head  # How many query heads per kv head

        # Fuse query, key, value projection
        self.qkv_proj = nn.Linear(
            n_embd, n_head * self.head_dim + 2 * n_kv_head * self.head_dim, bias=False
        )

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _emb_dim = x.size()

        # Project to query, key, value and split
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            (
                self.n_head * self.head_dim,
                self.n_kv_head * self.head_dim,
                self.n_kv_head * self.head_dim,
            ),
            dim=2,
        )
        # Reshape separate heads
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        # Repeat key and value to match query heads
        # (batch, seq_len, n_head, head_dim)

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention: (batch, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply scaled dot product attention
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transpose back and reshape: (batch, seq_len, n_embd)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)

        # Final output projection
        return self.c_proj(y)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat key/value tensors to match the number of query heads."""
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        if self.n_kv_head == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, num_kv_heads, self.num_query_groups, head_dim)
            .reshape(batch_size, seq_len, num_kv_heads * self.num_query_groups, head_dim)
        )


class CausalSelfAttentionGQA(nn.Module):
    """Causal self-attention leveraging grouped query attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # GQA configuration: use fewer kv heads
        # For GPT-2 small (12 heads), we can use 4 kv heads (3:1 ratio)
        # This gives good balance between performance and quality
        self.n_kv_head = config.n_kv_head

        self.gqa = GroupedQueryAttention(self.n_head, self.n_embd, self.n_kv_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gqa(x)
