"""Slow element-by-element reference, a literal transcription of the per-element formulas:

    H_t[i, j] = exp(delta_t[i] A[i, j]) H_{t-1}[i, j] + delta_t[i] B_t[j] x_t[i]
    y_t[i]    = sum_j C_t[j] H_t[i, j] + D[i] x_t[i]
"""

import math

import torch
from torch import Tensor


@torch.no_grad()
def naive_selective_scan(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]
    if initial_state is None:
        state = [[[0.0] * d_state for _ in range(d_inner)] for _ in range(batch)]
    else:
        state = initial_state.tolist()
    x_, delta_, A_, B_, C_, D_ = (t.tolist() for t in (x, delta, A, B, C, D))

    y = [[[0.0] * d_inner for _ in range(seq_len)] for _ in range(batch)]
    for b in range(batch):
        for t in range(seq_len):
            for i in range(d_inner):
                readout = 0.0
                for j in range(d_state):
                    a_bar = math.exp(delta_[b][t][i] * A_[i][j])
                    b_bar = delta_[b][t][i] * B_[b][t][j]
                    state[b][i][j] = a_bar * state[b][i][j] + b_bar * x_[b][t][i]
                    readout += C_[b][t][j] * state[b][i][j]
                y[b][t][i] = readout + D_[i] * x_[b][t][i]

    return (
        torch.tensor(y, dtype=x.dtype),
        torch.tensor(state, dtype=x.dtype),
    )
