"""Chunked SSD (State Space Duality) — pure PyTorch reference path.

Mirror of ``reference_repos/mamba/mamba_ssm/modules/ssd_minimal.py``.
You implement the bodies; signatures and contracts stay fixed for tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def segsum(x: Tensor) -> Tensor:
    """Stable lower-triangular segment sum in log-space.

    For the last axis of length ``T``, return ``S`` with:
        S[..., i, j] = sum_{k=j+1}^{i} x[..., k]   if i >= j
        S[..., i, j] = -inf                          if i < j

    (Equivalently: cumsum along a causal-shifted copy, then mask.)

    Args:
        x: ``(..., T)`` — typically log-decays ``A`` rearranged to ``(b, h, c, l)``.

    Returns:
        ``(..., T, T)`` segment sums (same dtype/device as ``x``).
    """
    X_cumsum = torch.cumsum(x, dim=-1)
    S = X_cumsum.unsqueeze(-1)- X_cumsum.unsqueeze(-2)
    strict_upper_mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)

    return S.masked_fill(strict_upper_mask, float("-inf"))


def ssd_sequential(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sequential recurrence reference implementation of discrete SSD."""
    b, L, h, p = X.shape
    assert L % block_len == 0, f"Sequence length {L} must be divisible by block_len {block_len}"
    n = B.shape[-1]

    ys = []
    if initial_states is None:
        states = torch.zeros((b, h, p, n), device=X.device, dtype=X.dtype)
    else:
        states = initial_states.squeeze(1).clone()

    for i in range(L):
        a_i = torch.exp(A[:, i, :])  # shape (b, h)
        b_i = B[:, i, :, :]  # shape (b, h, n)
        x_i = X[:, i, :, :]  # shape (b, h, p)
        c_i = C[:, i, :, :]  # shape (b, h, n)

        states = a_i.unsqueeze(-1).unsqueeze(-1) * states + torch.einsum("b h n, b h p -> b h p n", b_i, x_i)
        y_i = torch.einsum("b h p n, b h n -> b h p", states, c_i)
        ys.append(y_i)

    y = torch.stack(ys, dim=1)
    return y, states


def ssd_quadratic(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Quadratic (Attention-like) discrete SSD: ``Y = (L ⊙ (C B^T)) X``.

    Steps to implement:
        1. Validate ``L % block_len == 0``.
        2. Build causal decay mask ``L_mat = exp(segsum(A^T))`` of shape ``(b, h, L, L)``.
        3. Compute ``Y`` from ``(L_mat ⊙ (C B^T)) X`` (plus ``initial_states`` term if given).
        4. Compute ``final_state`` of shape ``(b, h, p, n)`` at ``t = L - 1``.
    """
    L = X.shape[-3]
    assert L % block_len == 0, f"Sequence length {L} must be divisible by block_len {block_len}"

    L_mat = torch.exp(segsum(A.transpose(-2, -1)))  # (..., h, L, L)
    CB_T = torch.einsum("... u h n, ... v h n -> ... h u v", C, B)  # (..., h, L, L)

    # Stream 1: output Y = (L \odot (C B^T)) X
    y = torch.einsum("... h u v, ... v h p -> ... u h p", L_mat * CB_T, X)

    # Stream 2: final_state at t = L - 1 reusing last row of L_mat
    final_states = torch.einsum("... l h n, ... l h p, ... h l -> ... h p n", B, X, L_mat[..., -1, :])

    if initial_states is not None:
        states = initial_states.squeeze(-4)  # (..., h, p, n)
        pref_A = torch.cumsum(A, dim=-2)  # (..., L, h)
        ## contribute of initial states to y and final_states
        y = y + torch.einsum("... l h n, ... h p n, ... l h -> ... l h p", C, states, torch.exp(pref_A))

        final_states = final_states + states * torch.exp(pref_A[..., -1, :, None, None])  # (..., h, p, n)

    return y, final_states


def ssd_chunked(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Chunked discrete SSD (Listing 1 / ``ssd_minimal_discrete``).

    Steps to implement (match the note / ref):
        1. Rearrange into chunks ``(b, c, l, ...)``.
        2. Intra-chunk diagonal: ``Y_diag`` via ``exp(segsum(A))``.
        3. Intra-chunk states at chunk ends.
        4. Inter-chunk state passing with chunk-level ``segsum``.
        5. ``Y_off`` from carried states; ``Y = Y_diag + Y_off``.
    """
    b, L, h, p = X.shape
    assert L % block_len == 0, f"Sequence length {L} must be divisible by block_len {block_len}"
    c = L // block_len

    # 1. Rearrange into chunks (b, c, l, ...)
    X_c, A_c, B_c, C_c = [t.reshape(b, c, block_len, *t.shape[2:]) for t in (X, A, B, C)]

    # 2. Intra-chunk diagonal (Y_diag) & chunk-end states (Right * X) via ssd_quadratic
    Y_diag, intra_states = ssd_quadratic(X_c, A_c, B_c, C_c, block_len=block_len, initial_states=None)

    # 3. Inter-chunk state passing
    if initial_states is None:
        initial_states = torch.zeros_like(intra_states[:, :1])
    states = torch.cat([initial_states, intra_states], dim=1)  # (b, c + 1, h, p, n)

    A_chunk_sum = A_c.sum(dim=2).transpose(1, 2)  # (b, h, c)
    decay_chunk = torch.exp(segsum(F.pad(A_chunk_sum, (1, 0))))  # (b, h, c + 1, c + 1)
    new_states = torch.einsum("b h z c, b c h p n -> b z h p n", decay_chunk, states)
    carried_states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Off-diagonal output (Left * carried_states) & combine
    Left = C_c * torch.exp(torch.cumsum(A_c, dim=2)).unsqueeze(-1)  # (b, c, l, h, n)
    Y_off = torch.einsum("b c l h n, b c h p n -> b c l h p", Left, carried_states)

    Y = (Y_diag + Y_off).reshape(b, L, h, p)
    return Y, final_state


# Register all implementations to test interchangeably via pytest (-k <name>).
SSD_IMPLS = {
    "sequential": ssd_sequential,
    "quadratic": ssd_quadratic,
    "chunked": ssd_chunked,
}


def ssd_minimal_discrete(
    X: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Default public entry point (delegates to ``ssd_chunked``).

    Args:
        X: ``(b, L, h, p)`` — value / input features per head.
        A: ``(b, L, h)`` — per-step log-decay (``delta * A_continuous``, already scaled).
        B: ``(b, L, h, n)`` — write vectors (after repeating groups to heads if needed).
        C: ``(b, L, h, n)`` — read vectors.
        block_len: chunk size ``l``; require ``L % block_len == 0``.
        initial_states: ``(b, 1, h, p, n)`` or None → zeros.

    Returns:
        Y: ``(b, L, h, p)``
        final_state: ``(b, h, p, n)``
    """
    return ssd_chunked(X, A, B, C, block_len, initial_states)
