import dataclasses
from typing import Literal


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int | None = 4

    activation_fn: Literal["gelu", "swiglu"] = "swiglu"


@dataclasses.dataclass
class QwenGPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int = 4
    head_dim: int = 64

    # Hybrid architecture: every 4th layer uses attention, others use linear
    attention_layer_indices: tuple[int, ...] = (3, 7, 11)  # 0-indexed: layers 4, 8, 12

    activation_fn: Literal["gelu", "swiglu"] = "swiglu"
    conv_kernel_size: int = 4
