from config import GPTConfig, QwenGPTConfig

num_vocab = 50257
MODEL_CONFIGS = {
    "baseline": GPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_head=None,
        activation_fn="gelu",
    ),  # 124M GPT-2
    "baseline-swiglu": GPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_head=None,
        activation_fn="swiglu",
    ),  # 124M GPT-2
    "d12": GPTConfig(
        vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, n_kv_head=4
    ),  # 124M GPT-2
    "d12-gelu": GPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_head=4,
        activation_fn="gelu",
    ),  # 124M GPT-2
    "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024, n_kv_head=4),
    "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280, n_kv_head=4),
    "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600, n_kv_head=5),
    "qwen": QwenGPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_head=4,
        head_dim=64,
        attention_layer_indices=(3, 7, 11),  # Every 4th layer
        activation_fn="swiglu",
        conv_kernel_size=4,
    ),  # 124M hybrid GPT-2 + Gated DeltaNet
}
