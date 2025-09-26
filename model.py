import math

import torch
import torch.nn as nn

from activation import SwiGLU
from attention import CausalSelfAttention
from baseline_mlp import MLP
from config import GPTConfig
from gqa_attention import CausalSelfAttentionGQA


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = (
            CausalSelfAttentionGQA(config) if config.n_kv_head else CausalSelfAttention(config)
        )
        self.swiglu = SwiGLU(config.n_embd, 4 * config.n_embd)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        return x + self.swiglu(rmsnorm(x))


class BaselineBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = (
            CausalSelfAttentionGQA(config) if config.n_kv_head else CausalSelfAttention(config)
        )
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        return x + self.mlp(rmsnorm(x))


# -----------------------------------------------------------------------------
# The main GPT-2 model


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        block_module = BaselineBlock if config.activation_fn == "gelu" else Block

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([block_module(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )


def rmsnorm(x0: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)
