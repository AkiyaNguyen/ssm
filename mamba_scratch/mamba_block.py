"""Mamba block (selective SSM wrapped with projections, causal conv and gating)."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mamba_scratch.ssm import selective_scan


@dataclass
class MambaConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    def __post_init__(self) -> None:
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


def init_A_log(d_inner: int, d_state: int) -> Tensor:
    """S4D-real initialization: A[i, j] = -(j + 1), so A_log[i, j] = log(j + 1).

    Returns:
        A_log: (d_inner, d_state), float32.
    """
    # TODO
    A_log_one_dim = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))  # (d_state,)
    A_log = A_log_one_dim.unsqueeze(0).repeat(d_inner, 1)  # (d_inner, d_state)
    return A_log

def init_dt_bias(
    d_inner: int,
    dt_min: float,
    dt_max: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Bias of dt_proj such that softplus(bias) ~ LogUniform[dt_min, dt_max].

    Steps:
        1. sample dt log-uniformly in [dt_min, dt_max]
        2. return the inverse softplus: dt + log(1 - exp(-dt))
           (use torch.expm1 for numerical stability)

    Returns:
        bias: (d_inner,)
    """
    def inverse_softplus(dt: Tensor) -> Tensor:
        return dt + torch.log(-torch.expm1(-dt))

    if generator is None:
        generator = torch.default_generator
    log_dt_min, log_dt_max = math.log(dt_min), math.log(dt_max)
    log_dt = torch.empty((d_inner,)).uniform_(log_dt_min, log_dt_max, generator=generator)  # (d_inner,)
    dt = torch.exp(log_dt)  #

    return inverse_softplus(dt)  # (d_inner,)

class MambaBlock(nn.Module):
    """(b, L, d_model) -> (b, L, d_model)."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        d_inner, d_state, dt_rank = config.d_inner, config.d_state, config.dt_rank

        self.in_proj = nn.Linear(config.d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=config.d_conv,
            groups=d_inner,
            padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False) # W_delta, B, C created  from input x
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True) #  to produce dt
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False) 

        self.A_log = nn.Parameter(init_A_log(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        with torch.no_grad():
            self.dt_proj.bias.copy_(init_dt_bias(d_inner, config.dt_min, config.dt_max))

    def selective_params(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Input-dependent SSM parameters.

        Args:
            x: (b, L, d_inner), output of conv + SiLU.

        Returns:
            delta: (b, L, d_inner) = softplus(dt_proj(W_delta x)), strictly positive
            B: (b, L, d_state)
            C: (b, L, d_state)
        """
        # TODO: split x_proj(x) into [dt_rank, d_state, d_state] along the last dim
        concat_params = self.x_proj(x)
        W_delta, B, C = torch.split(concat_params, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = torch.nn.Softplus()(self.dt_proj(W_delta))  # (b, L, d_inner)
        return delta, B, C

    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: (b, L, d_model)

        Returns:
            (b, L, d_model)
        """
        # TODO:
        #   1. x, z = in_proj(u) split in half                      (b, L, d_inner) each
        x, z = torch.chunk(self.in_proj(u), 2, dim=-1)

        #   2. x = SiLU(causal conv1d(x))  -- Conv1d expects (b, C, L); keep only the first L outputs
        x_conv = self.conv1d(x.transpose(1, 2))  # (b, d_inner, L + d_conv - 1)
        x = x_conv[..., :x.shape[1]].transpose(1, 2) # (b, L, d_inner)

        #   3. delta, B, C = self.selective_params(x)
        delta, B, C = self.selective_params(x)  # (b, L, d_inner), (b, L, d_state), (b, L, d_state)
        #   4. A = -exp(A_log)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        #   5. y = selective_scan(x, delta, A, B, C, D)
        y, _ = selective_scan(x, delta, A, B, C, self.D)  # (b, L, d_inner)
        #   6. return out_proj(y * SiLU(z))
        return self.out_proj(y * torch.nn.functional.silu(z))  # (b, L, d_model)

class ResidualMambaLayer(nn.Module):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model)
        self.mixer = MambaBlock(config)

    def forward(self, u: Tensor) -> Tensor:
        return u + self.mixer(self.norm(u))


class MambaLM(nn.Module):
    """Token ids (b, L) -> logits (b, L, vocab_size)."""

    def __init__(self, config: MambaConfig, vocab_size: int, n_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList(ResidualMambaLayer(config) for _ in range(n_layers))
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))
