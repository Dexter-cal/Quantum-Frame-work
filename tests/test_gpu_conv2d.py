"""v1.8.8 — qai.gpu.conv2d numpy fallback is a REAL convolution.

Regression tests for focus-list item 4: before v1.8.8 the numpy fallback
of qai.gpu.conv2d() returned the input tensor UNCHANGED (marked
"simplified", with an unused sliding_window_view import). Now it computes
a real vectorized cross-correlation (sliding_window_view + einsum) that
matches torch.nn.functional.conv2d semantics — which every test here
verifies against a naive loop reference.

These tests run the numpy path directly (the sandbox has no torch, so
`qai.gpu._torch is None` and every call goes through the fallback).
"""
import numpy as np
import pytest

from qai.gpu import conv2d, _conv2d_numpy


def _naive_conv2d(x, w, b=None, stride=1, padding=0):
    """Slow but obviously-correct reference: nested loops, float64."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]
    if w.ndim == 3:
        w = w[None] if w.shape[0] == x.shape[1] else w[:, None]
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    n, c, h, wd = x.shape
    o, wc, kh, kw = w.shape
    assert wc == c
    ho = (h - kh) // sh + 1
    wo = (wd - kw) // sw + 1
    out = np.zeros((n, o, ho, wo))
    for ni in range(n):
        for oi in range(o):
            for i in range(ho):
                for j in range(wo):
                    patch = x[ni, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    out[ni, oi, i, j] = np.sum(patch * w[oi])
    if b is not None:
        out += np.asarray(b, dtype=np.float64)[None, :, None, None]
    return out[0] if squeeze else out


def _rng():
    return np.random.default_rng(11)


# ---------------------------------------------------------------------------
# The core regression: output must actually be a convolution
# ---------------------------------------------------------------------------
def test_output_is_not_the_input():
    x = _rng().normal(size=(3, 8, 8)).astype(np.float32)
    w = _rng().normal(size=(4, 3, 3, 3)).astype(np.float32)
    out = conv2d(x, w)
    assert out.shape == (4, 6, 6)          # old bug returned (3, 8, 8) = x's shape


def test_matches_naive_reference_single_image():
    rng = _rng()
    x = rng.normal(size=(3, 9, 9))
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=(4,))
    for stride, padding in [(1, 0), (2, 0), (1, 1), (2, 1), (3, 2)]:
        got = conv2d(x, w, b, stride=stride, padding=padding)
        want = _naive_conv2d(x, w, b, stride=stride, padding=padding)
        assert got.shape == want.shape, (stride, padding)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_matches_naive_reference_batch():
    rng = _rng()
    x = rng.normal(size=(2, 3, 10, 7))
    w = rng.normal(size=(5, 3, 3, 2))       # asymmetric kernel
    got = conv2d(x, w, stride=(2, 1), padding=(1, 0))
    want = _naive_conv2d(x, w, stride=(2, 1), padding=(1, 0))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_1x1_kernel_is_linear_projection():
    # (1,1) kernel with C_out == C_in is a per-pixel linear map: windows ==
    # pixels, so the naive reference and the implementation must agree
    # exactly — this also pins the cross-correlation (no kernel flip).
    rng = _rng()
    x = rng.normal(size=(2, 3, 5, 5))
    w = rng.normal(size=(3, 3, 1, 1))
    got = conv2d(x, w)
    want = np.einsum("nchw,oc->nohw", x, w[:, :, 0, 0])
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_identity_kernel_returns_input():
    # identity kernel (center 1) with matching padding reproduces the input
    x = _rng().normal(size=(1, 5, 5)).astype(np.float32)
    w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    w[0, 0, 1, 1] = 1.0
    out = conv2d(x, w, padding=1)
    np.testing.assert_allclose(out, x, rtol=1e-5)


# ---------------------------------------------------------------------------
# shapes, dtypes, bias
# ---------------------------------------------------------------------------
def test_output_shape_formula():
    rng = _rng()
    x = rng.normal(size=(2, 4, 11, 13))
    w = rng.normal(size=(6, 4, 3, 5))
    for stride, padding, want in [
        (1, 0, (2, 6, 9, 9)),
        (2, 1, (2, 6, 6, 6)),
        ((1, 2), (0, 1), (2, 6, 9, 6)),
    ]:
        got = conv2d(x, w, stride=stride, padding=padding)
        assert got.shape == want, (stride, padding, got.shape)


def test_bias_added_per_channel():
    rng = _rng()
    x = rng.normal(size=(3, 6, 6))
    w = rng.normal(size=(2, 3, 3, 3))
    b = np.array([10.0, -10.0], dtype=np.float32)
    with_b = conv2d(x, w, b)
    without = conv2d(x, w)
    np.testing.assert_allclose(with_b[0] - without[0], 10.0, atol=1e-4)
    np.testing.assert_allclose(with_b[1] - without[1], -10.0, atol=1e-4)


def test_output_dtype_float32():
    x = _rng().normal(size=(1, 4, 4)).astype(np.float64)  # even float64 in
    w = _rng().normal(size=(2, 1, 2, 2)).astype(np.float64)
    out = conv2d(x, w)
    assert out.dtype == np.float32


def test_3d_weight_single_output_kernel():
    # (C_in, kH, kW) with first dim == C_in -> one output channel
    rng = _rng()
    x = rng.normal(size=(3, 7, 7))
    w = rng.normal(size=(3, 3, 3))
    out = conv2d(x, w)
    assert out.shape == (1, 5, 5)
    want = _naive_conv2d(x, w)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)


def test_3d_weight_multi_output_c1_kernel():
    # (C_out, kH, kW) with C_in == 1 -> multi-output, single-channel kernel
    rng = _rng()
    x = rng.normal(size=(1, 7, 7))
    w = rng.normal(size=(4, 3, 3))
    out = conv2d(x, w)
    assert out.shape == (4, 5, 5)
    want = _naive_conv2d(x, w)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# honest errors
# ---------------------------------------------------------------------------
def test_channel_mismatch_raises():
    x = _rng().normal(size=(3, 8, 8))
    w = _rng().normal(size=(4, 2, 3, 3))   # expects 2 channels, got 3
    with pytest.raises(ValueError, match="channel mismatch"):
        conv2d(x, w)


def test_kernel_larger_than_input_raises():
    x = _rng().normal(size=(1, 3, 3))
    w = _rng().normal(size=(1, 1, 5, 5))
    with pytest.raises(ValueError, match="larger than"):
        conv2d(x, w)


def test_bad_input_ndim_raises():
    with pytest.raises(ValueError, match="conv2d received input"):
        conv2d(np.zeros((8, 8)), np.zeros((1, 1, 3, 3)))
    with pytest.raises(ValueError, match="conv2d received weight"):
        conv2d(np.zeros((1, 8, 8)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# torch parity (skipped where torch is absent — always, in this sandbox)
# ---------------------------------------------------------------------------
def test_torch_parity_if_available():
    torch = pytest.importorskip("torch")
    rng = _rng()
    x = rng.normal(size=(2, 3, 9, 9)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=(4,)).astype(np.float32)
    for stride, padding in [(1, 0), (2, 1)]:
        got = conv2d(x, w, b, stride=stride, padding=padding)
        want = torch.nn.functional.conv2d(
            torch.from_numpy(x), torch.from_numpy(w),
            bias=torch.from_numpy(b), stride=stride, padding=padding,
        ).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
