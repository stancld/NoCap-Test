import torch

from train_gpt2 import Block, GPTConfig


def test_vanilla_block():
    config = GPTConfig(vocab_size=100, n_layer=12, n_head=12, n_embd=768)
    block = Block(config)

    x = torch.randn(1, 128, 768)
    block(x)
