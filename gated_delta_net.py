import torch
import torch.nn as nn


class GatedDeltaNet(nn.Module):
    """Gated Delta Net recently used in Qwen 3 Next.

    References
    ----------
    Gated Delta Networks: Improving Mamba2 with Delta Rule
        https://arxiv.org/pdf/2412.06464
    """

    def __init__(
        self,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        n_embd: int,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()

        if num_key_heads != num_value_heads:
            raise NotImplementedError(
                f"num_key_heads ({num_key_heads}) must equal num_value_heads ({num_value_heads})"
            )

        self.num_key_heads = num_key_heads
        self.key_head_dim = key_head_dim
        self.key_dim = num_key_heads * key_head_dim

        self.num_value_heads = num_value_heads
        self.value_head_dim = value_head_dim
        self.value_dim = num_value_heads * value_head_dim

        self.n_embd = n_embd
        self.conv_kernel_size = conv_kernel_size

        self.qkv_proj = nn.Linear(n_embd, 2 * self.key_dim + self.value_dim, bias=False)

        # Update & forget gate (beta for values, alpha for keys)
        self.delta_gates_ba = nn.Linear(n_embd, self.value_dim + self.key_dim, bias=False)
        self.output_gate = nn.Linear(n_embd, self.value_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, n_embd, bias=False)

        # QKV 1D conv (causal)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,  # We'll handle causal padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _emb_dim = x.size()

        qkv = self.qkv_proj(x)
        # Apply causal convolution
        qkv_conv_input = qkv.transpose(1, 2)  # [batch, conv_dim, seq_len]
        # Add left padding for causal convolution
        qkv_padded = nn.functional.pad(qkv_conv_input, (self.conv_kernel_size - 1, 0))
        qkv = nn.functional.silu(self.conv1d(qkv_padded))
        qkv = qkv.transpose(1, 2)

        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=2)

        q = q.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)

        beta_update_gate, alpha_forget_gate = self.delta_gates_ba(x).split(
            (self.value_dim, self.key_dim), dim=2
        )
        output_gate = self.output_gate(x)

        # Apply activations to gates
        alpha_forget_gate = torch.sigmoid(alpha_forget_gate)
        beta_update_gate = torch.sigmoid(beta_update_gate)

        # Reshape gates to match head dimensions
        alpha_forget_gate = alpha_forget_gate.reshape(
            batch_size, seq_len, self.num_key_heads, self.key_head_dim
        )
        beta_update_gate = beta_update_gate.reshape(
            batch_size, seq_len, self.num_value_heads, self.value_head_dim
        )
        output_gate = output_gate.reshape(
            batch_size, seq_len, self.num_value_heads, self.value_head_dim
        )

        attn_out = self.torch_recurrent_gated_delta_rule(
            q, k, v, alpha_forget_gate, beta_update_gate
        )

        # Apply output gate and project back to embedding space
        gated_out = attn_out * torch.sigmoid(output_gate)
        gated_out = gated_out.reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(gated_out)

    def torch_recurrent_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
            query
                [batch, seq_len, num_key_heads, key_head_dim]
            key
                [batch, seq_len, num_key_heads, key_head_dim]
            value
                [batch, seq_len, num_value_heads, value_head_dim]
            alpha
                [batch, seq_len, num_key_heads, key_head_dim] - forget gate
            beta
                [batch, seq_len, num_value_heads, value_head_dim] - update gate

        Returns
        --------
            output
                [batch, seq_len, num_value_heads, value_head_dim]
        """
        batch_size, seq_len, num_key_heads, key_head_dim = query.shape
        _num_value_heads, value_head_dim = value.shape[2], value.shape[3]

        # Transpose to [batch, num_heads, seq_len, head_dim] for processing
        query = query.transpose(1, 2)  # [batch, num_key_heads, seq_len, key_head_dim]
        key = key.transpose(1, 2)  # [batch, num_key_heads, seq_len, key_head_dim]
        value = value.transpose(1, 2)  # [batch, num_value_heads, seq_len, value_head_dim]
        alpha = alpha.transpose(1, 2)  # [batch, num_key_heads, seq_len, key_head_dim]
        beta = beta.transpose(1, 2)  # [batch, num_value_heads, seq_len, value_head_dim]

        # Scale query
        scale = 1.0 / (key_head_dim**0.5)
        query = query * scale

        # Initialize output and recurrent state
        output = torch.zeros_like(value)
        recurrent_state = torch.zeros(
            batch_size,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            device=value.device,
            dtype=value.dtype,
        )

        # Recurrent processing
        for t in range(seq_len):
            q_t = query[:, :, t]  # [batch, num_key_heads, key_head_dim]
            k_t = key[:, :, t]  # [batch, num_key_heads, key_head_dim]
            v_t = value[:, :, t]  # [batch, num_value_heads, value_head_dim]
            alpha_t = alpha[:, :, t]  # [batch, num_key_heads, key_head_dim]
            beta_t = beta[:, :, t]  # [batch, num_value_heads, value_head_dim]

            # Apply forget gate to recurrent state
            alpha_gate = alpha_t.unsqueeze(-1)  # [batch, num_key_heads, key_head_dim, 1]
            recurrent_state *= alpha_gate

            # Compute predicted value from memory
            kv_mem = torch.einsum("bhk,bhkv->bhv", k_t, recurrent_state)

            # Compute delta update
            delta = (v_t - kv_mem) * beta_t  # [batch, num_value_heads, value_head_dim]

            # Update recurrent state
            state_update = torch.einsum("bhk,bhv->bhkv", k_t, delta)
            recurrent_state = recurrent_state + state_update
            # Compute output
            output[:, :, t] = torch.einsum("bhk,bhkv->bhv", q_t, recurrent_state)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        return output.transpose(1, 2)

    def _reshape_gate_states(
        self, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = a.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        b = b.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        z = z.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        return a, b, z
