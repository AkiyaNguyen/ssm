import pytest
import torch
import torch.nn.functional as F

from mamba_scratch.ssm import discretize, selective_scan
from tests.reference import naive_selective_scan

BATCH, SEQ_LEN, D_INNER, D_STATE = 2, 7, 3, 4


@pytest.fixture
def ssm_inputs():
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return {
        "x": randn(BATCH, SEQ_LEN, D_INNER),
        "delta": F.softplus(randn(BATCH, SEQ_LEN, D_INNER)),
        "A": -torch.exp(randn(D_INNER, D_STATE)),
        "B": randn(BATCH, SEQ_LEN, D_STATE),
        "C": randn(BATCH, SEQ_LEN, D_STATE),
        "D": randn(D_INNER),
    }


# ---------- discretize ----------


def test_discretize_shapes(ssm_inputs):
    A_bar, B_bar = discretize(ssm_inputs["delta"], ssm_inputs["A"], ssm_inputs["B"])
    assert A_bar.shape == (BATCH, SEQ_LEN, D_INNER, D_STATE)
    assert B_bar.shape == (BATCH, SEQ_LEN, D_INNER, D_STATE)


def test_discretize_matches_elementwise_definition(ssm_inputs):
    delta, A, B = ssm_inputs["delta"], ssm_inputs["A"], ssm_inputs["B"]
    A_bar, B_bar = discretize(delta, A, B)
    for i in range(D_INNER):
        for j in range(D_STATE):
            torch.testing.assert_close(A_bar[..., i, j], torch.exp(delta[..., i] * A[i, j]))
            torch.testing.assert_close(B_bar[..., i, j], delta[..., i] * B[..., j])


def test_A_bar_lies_in_unit_interval(ssm_inputs):
    A_bar, _ = discretize(ssm_inputs["delta"], ssm_inputs["A"], ssm_inputs["B"])
    assert torch.all(A_bar > 0) and torch.all(A_bar < 1)


# ---------- selective_scan ----------


def test_selective_scan_shapes(ssm_inputs):
    y, final_state = selective_scan(**ssm_inputs)
    assert y.shape == (BATCH, SEQ_LEN, D_INNER)
    assert final_state.shape == (BATCH, D_INNER, D_STATE)


def test_selective_scan_matches_naive_reference(ssm_inputs):
    y, final_state = selective_scan(**ssm_inputs)
    y_ref, final_state_ref = naive_selective_scan(**ssm_inputs)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(final_state, final_state_ref)


def test_selective_scan_with_initial_state_matches_reference(ssm_inputs):
    initial_state = torch.randn(BATCH, D_INNER, D_STATE, dtype=torch.float64)
    y, final_state = selective_scan(**ssm_inputs, initial_state=initial_state)
    y_ref, final_state_ref = naive_selective_scan(**ssm_inputs, initial_state=initial_state)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(final_state, final_state_ref)


def test_zero_delta_keeps_state_and_leaves_only_skip(ssm_inputs):
    """delta = 0 => A_bar = 1, B_bar = 0 => state never changes, y = D * x."""
    ssm_inputs["delta"] = torch.zeros_like(ssm_inputs["delta"])
    y, final_state = selective_scan(**ssm_inputs)
    torch.testing.assert_close(y, ssm_inputs["D"] * ssm_inputs["x"])
    torch.testing.assert_close(final_state, torch.zeros_like(final_state))


def test_splitting_sequence_and_passing_state_equals_full_scan(ssm_inputs):
    split = SEQ_LEN // 2
    first = {k: v[:, :split] if v.dim() == 3 else v for k, v in ssm_inputs.items()}
    second = {k: v[:, split:] if v.dim() == 3 else v for k, v in ssm_inputs.items()}

    y_full, state_full = selective_scan(**ssm_inputs)
    y_first, state_mid = selective_scan(**first)
    y_second, state_end = selective_scan(**second, initial_state=state_mid)

    torch.testing.assert_close(torch.cat([y_first, y_second], dim=1), y_full)
    torch.testing.assert_close(state_end, state_full)


def test_lti_case_equals_convolution(ssm_inputs):
    """Constant delta, B, C (no selectivity) => y = causal conv of x with
    K_k[i] = sum_j C[j] * A_bar[i, j]^k * delta[i] * B[j], plus D * x."""
    x, A, D = ssm_inputs["x"], ssm_inputs["A"], ssm_inputs["D"]
    delta_const = F.softplus(torch.randn(D_INNER, dtype=torch.float64))
    B_const = torch.randn(D_STATE, dtype=torch.float64)
    C_const = torch.randn(D_STATE, dtype=torch.float64)

    y, _ = selective_scan(
        x,
        delta_const.expand(BATCH, SEQ_LEN, D_INNER),
        A,
        B_const.expand(BATCH, SEQ_LEN, D_STATE),
        C_const.expand(BATCH, SEQ_LEN, D_STATE),
        D,
    )

    A_bar = torch.exp(delta_const[:, None] * A)  # (d, n)
    lags = torch.arange(SEQ_LEN, dtype=torch.float64)
    kernel = torch.einsum(
        "j,kij,i,j->ki", C_const, A_bar[None] ** lags[:, None, None], delta_const, B_const
    )  # (L, d)
    y_conv = torch.zeros_like(x)
    for t in range(SEQ_LEN):
        for k in range(t + 1):
            y_conv[:, t] += kernel[k] * x[:, t - k]
    y_conv += D * x

    torch.testing.assert_close(y, y_conv)


def test_gradients_flow_to_all_inputs(ssm_inputs):
    leaves = {k: v.clone().requires_grad_(True) for k, v in ssm_inputs.items()}
    y, _ = selective_scan(**leaves)
    y.sum().backward()
    for name, tensor in leaves.items():
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
