import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Standard SwiGLU implementation.

    Source
    ------
    Noam Shazeer's "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        # The paper recommends the hidden dimension be 2/3 of the FFN dimension
        hidden_dim = int(2 * d_ffn / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        return self.w3(gate * self.w2(x))
