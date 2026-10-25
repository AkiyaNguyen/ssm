"""Selective SSM (S6) core: discretization + sequential scan.

Shape convention (see Notes/ssm-notes/mamba-selective-ssm.md, section 4.1):
    b = batch, L = sequence length, d = d_inner (number of channels), n = d_state.
"""

import torch
from torch import Tensor


def discretize(delta: Tensor, A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Discretize the continuous parameters with step size ``delta``.

    A_bar uses zero-order hold, B_bar uses the Euler approximation:
        A_bar[..., i, j] = exp(delta[..., i] * A[i, j])
        B_bar[..., i, j] = delta[..., i] * B[..., j]

    Args:
        delta: (b, L, d), positive step sizes.
        A: (d, n), negative diagonal entries of each channel's state matrix.
        B: (b, L, n), input-dependent write direction, shared across channels.

    Returns:
        A_bar: (b, L, d, n)
        B_bar: (b, L, d, n)
    """
    # TODO: broadcast delta over n and B over d; no loops needed.
    A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (b, L, d, n)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)  # (b, L, d, n)

    return A_bar, B_bar


def selective_scan(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the selective SSM recurrence sequentially over time.

    For every step t:
        H_t = A_bar_t * H_{t-1} + (delta_t * x_t) B_t^T      (d, n)
        y_t = H_t C_t + D * x_t                              (d,)

    Args:
        x: (b, L, d), input of each channel.
        delta: (b, L, d)
        A: (d, n)
        B: (b, L, n)
        C: (b, L, n), input-dependent read direction, shared across channels.
        D: (d,), skip connection.
        initial_state: (b, d, n) or None (zeros).

    Returns:
        y: (b, L, d)
        final_state: (b, d, n), the state H_L after the last step.
    """
    # TODO:
    b, L, d = x.shape
    n = B.shape[-1]
    #   1. A_bar, B_bar = discretize(delta, A, B)
    A_bar, B_bar = discretize(delta, A, B)  # (b, L, d, n), (b, L, d, n)

    #   2. h = initial_state, or zeros of shape (b, d, n)
    h = initial_state if initial_state is not None \
        else torch.zeros(b, d, n, device=x.device, dtype=x.dtype)
    
    #   3. loop over t: update h, read out y_t with C_t
    y = []
    for t in range(L):
        A_bar_t = A_bar[:, t, :, :] # (b, d, n)
        B_bar_t = B_bar[:, t, :, :] # (b, d, n)
        C_t = C[:, t, :] # (b, n)
        x_t = x[:, t, :] # (b, d)

        new_h = A_bar_t * h + B_bar_t * x_t.unsqueeze(-1)
        y_t = D * x_t + torch.sum(new_h * C_t[:, None, :], dim=-1)
        # update 
        y.append(y_t)
        h = new_h       
    return torch.stack(y, dim=1), h
