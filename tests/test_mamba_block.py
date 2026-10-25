import math

import pytest
import torch
import torch.nn.functional as F

from mamba_scratch.mamba_block import (
    MambaBlock,
    MambaConfig,
    MambaLM,
    init_A_log,
    init_dt_bias,
)

BATCH, SEQ_LEN = 2, 10


@pytest.fixture
def config():
    return MambaConfig(d_model=32, d_state=8)


@pytest.fixture
def block(config):
    torch.manual_seed(0)
    return MambaBlock(config).double()


def random_input(config, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, config.d_model, generator=generator, dtype=torch.float64)


# ---------- config & init ----------


def test_config_derived_sizes(config):
    assert config.d_inner == 2 * config.d_model
    assert config.dt_rank == math.ceil(config.d_model / 16)


def test_init_A_log_gives_s4d_real():
    d_inner, d_state = 5, 6
    A_log = init_A_log(d_inner, d_state)
    assert A_log.shape == (d_inner, d_state)
    expected_A = -torch.arange(1, d_state + 1, dtype=torch.float32).expand(d_inner, d_state)
    torch.testing.assert_close(-torch.exp(A_log), expected_A)


def test_init_dt_bias_softplus_lies_in_range():
    dt_min, dt_max = 1e-3, 1e-1
    bias = init_dt_bias(4096, dt_min, dt_max, generator=torch.Generator().manual_seed(0))
    assert bias.shape == (4096,)
    dt = F.softplus(bias)
    assert dt.min() >= dt_min * (1 - 1e-4)
    assert dt.max() <= dt_max * (1 + 1e-4)
    log_dt = torch.log(dt)
    midpoint = (math.log(dt_min) + math.log(dt_max)) / 2
    assert abs(log_dt.mean().item() - midpoint) < 0.1, "dt should be log-uniform"


# ---------- selective params ----------


def test_selective_params_shapes_and_positive_delta(block, config):
    x = torch.randn(BATCH, SEQ_LEN, config.d_inner, dtype=torch.float64)
    delta, B, C = block.selective_params(x)
    assert delta.shape == (BATCH, SEQ_LEN, config.d_inner)
    assert B.shape == (BATCH, SEQ_LEN, config.d_state)
    assert C.shape == (BATCH, SEQ_LEN, config.d_state)
    assert torch.all(delta > 0)


def test_selective_params_depend_on_input(block, config):
    x1 = torch.randn(BATCH, SEQ_LEN, config.d_inner, dtype=torch.float64)
    x2 = torch.randn(BATCH, SEQ_LEN, config.d_inner, dtype=torch.float64)
    for p1, p2 in zip(block.selective_params(x1), block.selective_params(x2)):
        assert not torch.allclose(p1, p2)


# ---------- block ----------


def test_block_output_shape(block, config):
    out = block(random_input(config))
    assert out.shape == (BATCH, SEQ_LEN, config.d_model)


def test_block_is_causal(block, config):
    """Changing tokens at positions >= cut must not change outputs before cut."""
    cut = SEQ_LEN // 2
    u = random_input(config, seed=1)
    u_perturbed = u.clone()
    u_perturbed[:, cut:] = random_input(config, seed=2)[:, cut:]

    out, out_perturbed = block(u), block(u_perturbed)
    torch.testing.assert_close(out[:, :cut], out_perturbed[:, :cut])
    assert not torch.allclose(out[:, cut:], out_perturbed[:, cut:])


def test_all_block_parameters_receive_gradients(block, config):
    block(random_input(config)).pow(2).sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} is not used in forward"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


# ---------- language model ----------


def test_language_model_logits_shape(config):
    vocab_size = 50
    model = MambaLM(config, vocab_size=vocab_size, n_layers=2)
    tokens = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    logits = model(tokens)
    assert logits.shape == (BATCH, SEQ_LEN, vocab_size)
    assert torch.isfinite(logits).all()
