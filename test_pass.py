import pytest
import torch

from config import GPTConfig
from gated_delta_net import GatedDeltaNet
from gqa_attention import GroupedQueryAttention
from model import BaselineBlock, Block

HIDDEN_DIM = 768


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(2, 128, HIDDEN_DIM)


def test_gqa(x: torch.Tensor) -> None:
    gqa = GroupedQueryAttention(12, HIDDEN_DIM, 4)
    gqa(x)


def test_baseline_block(x: torch.Tensor) -> None:
    config = GPTConfig(
        vocab_size=100,
        n_layer=12,
        n_head=12,
        n_embd=HIDDEN_DIM,
        n_kv_head=None,
        activation_fn="gelu",
    )
    block = BaselineBlock(config)

    block(x)


def test_vanilla_block(x: torch.Tensor) -> None:
    config = GPTConfig(vocab_size=100, n_layer=12, n_head=12, n_embd=HIDDEN_DIM, n_kv_head=None)
    block = Block(config)

    block(x)


def test_gqa_block(x: torch.Tensor) -> None:
    config = GPTConfig(vocab_size=100, n_layer=12, n_head=12, n_embd=768, n_kv_head=4)
    block = Block(config)

    block(x)


def test_gdn(x: torch.Tensor) -> None:
    gdn = GatedDeltaNet(
        num_key_heads=2, num_value_heads=2, key_head_dim=16, value_head_dim=32, n_embd=768
    )

    gdn(x)
