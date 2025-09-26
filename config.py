import dataclasses


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int | None = 4
