"""Round-trip tests for the sm_70-safe INT4 quantize/dequantize wrappers.

These exist because the original ``test_quantize.py`` exercises the sm_80+
kernel that emits ``cvt.bf16.f32`` PTX and cannot run on Volta. The sm_70
wrappers route bf16 inputs through fp32 (work + scales) so we want to verify
the round-trip is stable for the cases that were silently broken in the
earlier fp16-scales draft (tiny-magnitude groups and outlier values).
"""

from __future__ import annotations

import math

import pytest
import torch

from kernels.llama3.quantize_sm70 import (
    _dequantize_int4_torch,
    _quantize_int4_torch,
    dequantize_int4,
    quantize_int4,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _max_abs(x: torch.Tensor) -> float:
    return float(x.float().abs().max().item())


@cuda
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_sm70_round_trip_bf16(group_size: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16)

    packed, scales = quantize_int4(x, group_size=group_size)
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float32
    assert scales.shape[-1] == x.shape[-1] // group_size

    x_hat = dequantize_int4(
        packed, scales, group_size=group_size, d_head=x.shape[-1], out_dtype=torch.bfloat16
    )
    assert x_hat.dtype == torch.bfloat16
    assert torch.isfinite(x_hat).all()

    err = (x.float() - x_hat.float()).abs().mean().item()
    scale_mean = scales.float().abs().mean().item()
    assert err < scale_mean, f"mean abs err {err} >= mean scale {scale_mean}"


@cuda
def test_sm70_round_trip_fp16() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 4, 32, 128, device="cuda", dtype=torch.float16)

    packed, scales = quantize_int4(x, group_size=32)
    assert scales.dtype == torch.float32

    x_hat = dequantize_int4(packed, scales, group_size=32, d_head=128, out_dtype=torch.float16)
    assert x_hat.dtype == torch.float16
    assert torch.isfinite(x_hat).all()

    err = (x.float() - x_hat.float()).abs().mean().item()
    scale_mean = scales.float().abs().mean().item()
    assert err < scale_mean


@cuda
def test_sm70_round_trip_tiny_magnitudes() -> None:
    """Groups with max_abs around 1e-6 used to underflow fp16 scales to zero."""
    torch.manual_seed(2)
    x = (torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float32) * 1e-6).to(torch.bfloat16)

    packed, scales = quantize_int4(x, group_size=32)
    assert scales.dtype == torch.float32
    # All scales should be strictly positive and finite (no underflow to 0).
    assert torch.isfinite(scales).all()
    assert (scales > 0).all()

    x_hat = dequantize_int4(
        packed, scales, group_size=32, d_head=128, out_dtype=torch.bfloat16
    )
    assert torch.isfinite(x_hat).all()

    # Compare against the torch reference round-trip on the same input.
    packed_t, scales_t = _quantize_int4_torch(x, group_size=32)
    x_hat_t = _dequantize_int4_torch(
        packed_t, scales_t, group_size=32, d_head=128, out_dtype=torch.bfloat16
    )
    diff = (x_hat.float() - x_hat_t.float()).abs().mean().item()
    ref_scale = scales_t.float().abs().mean().item()
    assert diff < ref_scale, f"triton vs torch tiny-magnitude diff {diff} >= ref scale {ref_scale}"


@cuda
def test_sm70_round_trip_large_magnitudes() -> None:
    """Outlier channels near the fp16 limit must not produce Inf/NaN."""
    torch.manual_seed(3)
    base = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float32)
    # Inject a few outlier columns at ~5000 magnitude.
    base[..., ::32] = base[..., ::32] * 5000.0
    x = base.to(torch.bfloat16)

    packed, scales = quantize_int4(x, group_size=32)
    assert torch.isfinite(scales).all()

    x_hat = dequantize_int4(
        packed, scales, group_size=32, d_head=128, out_dtype=torch.bfloat16
    )
    assert torch.isfinite(x_hat).all()

    # Bound on relative error: INT4 groupwise gives mean abs err <= scale_mean.
    err = (x.float() - x_hat.float()).abs().mean().item()
    scale_mean = scales.float().abs().mean().item()
    assert err < scale_mean, f"mean abs err {err} >= mean scale {scale_mean}"
    # And no single value should be wildly off (sanity bound).
    rel = (x.float() - x_hat.float()).abs().max().item() / max(_max_abs(x), 1.0)
    assert rel < 0.2, f"max relative err {rel} too large"


@cuda
def test_sm70_triton_matches_torch_bf16() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 8, 32, 128, device="cuda", dtype=torch.bfloat16)

    packed_t, scales_t = _quantize_int4_torch(x, group_size=32)
    packed_k, scales_k = quantize_int4(x, group_size=32)
    assert scales_t.dtype == torch.float32
    assert scales_k.dtype == torch.float32
    assert torch.allclose(scales_t, scales_k, rtol=1e-3, atol=1e-3)

    x_hat_t = _dequantize_int4_torch(
        packed_t, scales_t, group_size=32, d_head=128, out_dtype=torch.bfloat16
    )
    x_hat_k = dequantize_int4(packed_k, scales_k, group_size=32, d_head=128, out_dtype=torch.bfloat16)
    diff = (x_hat_t.float() - x_hat_k.float()).abs().mean().item()
    scale_mean = scales_k.float().abs().mean().item()
    assert diff < scale_mean * 0.1, (
        f"triton vs torch mean abs diff {diff} too large vs scale {scale_mean}"
    )


@cuda
def test_sm70_concat_along_seq_matches_one_shot() -> None:
    """Same layout as cache: cat packed/scales on seq dim after independent quantize."""
    torch.manual_seed(5)
    group_size = 32
    x = torch.randn(1, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
    split = 37
    a = x[:, :, :split, :].contiguous()
    b = x[:, :, split:, :].contiguous()
    pa, sa = quantize_int4(a, group_size=group_size)
    pb, sb = quantize_int4(b, group_size=group_size)
    pc = torch.cat([pa, pb], dim=2)
    sc = torch.cat([sa, sb], dim=2)
    d_head = int(sc.shape[-1] * group_size)
    x_cat = dequantize_int4(
        pc, sc, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16
    )
    p0, s0 = quantize_int4(x, group_size=group_size)
    x0 = dequantize_int4(
        p0, s0, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16
    )
    diff = (x_cat.float() - x0.float()).abs().max().item()
    scale_m = s0.float().abs().mean().item()
    assert diff < scale_m * 2.0, f"concat vs one-shot max_abs {diff} vs scale mean {scale_m}"


def _realistic_kv_like(
    *, batch: int, n_kv: int, seq: int, d_head: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """Synthesize K/V with per-head scale variance and sparse outliers.

    Shape mirrors the cache layout ``[B, n_kv, S, d_head]``. Per-head magnitudes
    span ~2 orders of magnitude and a few outlier channels are boosted so group
    scales pick up a wide dynamic range. Peak magnitude is capped to ~6e3, well
    inside both bf16 and fp16 representable range, so the test input itself is
    not the failure source; any non-finite output comes from quant/dequant.
    """
    torch.manual_seed(11)
    base = torch.randn(batch, n_kv, seq, d_head, device=device, dtype=torch.float32)
    head_scales = torch.logspace(0, math.log10(20.0), n_kv, device=device, dtype=torch.float32)
    base = base * head_scales.view(1, n_kv, 1, 1)
    outlier_channels = torch.arange(0, d_head, max(d_head // 8, 1), device=device)
    base[..., outlier_channels] *= 100.0
    return base.to(dtype)


@cuda
def test_sm70_realistic_kv_concat_torch_stays_finite() -> None:
    """Concat-on-seq quant+dequant through the torch path must stay finite.

    This covers the Volta mitigation path (``KV_INT4_TRITON=0`` / sm<8 autodetect):
    on a realistic-magnitude input with per-head scale variance and outlier
    channels, the fp32-only torch implementation must round-trip without nan/inf.
    The bug it guards against is a packed/scales width drift after ``torch.cat``
    or a scale underflow that silently corrupts a group.
    """
    from kernels.llama3.quantize_sm70 import (
        _dequantize_int4_torch,
        _quantize_int4_torch,
    )

    group_size = 32
    dtype = torch.bfloat16
    x = _realistic_kv_like(
        batch=1, n_kv=8, seq=96, d_head=128, device="cuda", dtype=dtype
    )
    split = 37
    a = x[:, :, :split, :].contiguous()
    b = x[:, :, split:, :].contiguous()
    pa, sa = _quantize_int4_torch(a, group_size=group_size)
    pb, sb = _quantize_int4_torch(b, group_size=group_size)
    assert torch.isfinite(sa).all(), "chunk-a scales non-finite"
    assert torch.isfinite(sb).all(), "chunk-b scales non-finite"
    pc = torch.cat([pa, pb], dim=2)
    sc = torch.cat([sa, sb], dim=2)
    assert pc.shape[-1] == sc.shape[-1] * (group_size // 2), (
        "packed/scales width invariant broken after cat"
    )
    d_head = int(sc.shape[-1] * group_size)
    x_hat = _dequantize_int4_torch(
        pc, sc, group_size=group_size, d_head=d_head, out_dtype=dtype
    )
    nan_n = int(torch.isnan(x_hat.float()).sum().item())
    inf_n = int(torch.isinf(x_hat.float()).sum().item())
    assert nan_n == 0 and inf_n == 0, f"torch dequant produced nan={nan_n} inf={inf_n}"

    # Sanity: concat-then-dequant should agree with one-shot dequant.
    p0, s0 = _quantize_int4_torch(x, group_size=group_size)
    x0 = _dequantize_int4_torch(
        p0, s0, group_size=group_size, d_head=d_head, out_dtype=dtype
    )
    scale_m = s0.float().abs().mean().item()
    diff = (x_hat.float() - x0.float()).abs().max().item()
    assert diff < scale_m * 2.0, (
        f"concat vs one-shot max_abs={diff} scale_mean={scale_m}"
    )


@cuda
def test_sm70_triton_matches_torch_within_fp16_range() -> None:
    """On fp16-safe realistic magnitudes, Triton and torch dequant agree and stay finite.

    "fp16-safe" matters because the sm_70 Triton dequant writes into an fp16
    intermediate buffer (cvt.f32.bf16 is illegal on Volta), so dequantized
    magnitudes above ~65504 overflow to +/-inf. This test keeps max|x| well inside
    fp16 range so it validates the kernel arithmetic itself, not the buffer cast.
    See ``test_sm70_triton_overflows_on_fp16_out_of_range`` for the documented
    overflow edge.
    """
    from kernels.llama3.quantize_sm70 import (
        _dequantize_int4_torch,
        _dequantize_int4_triton,
        _quantize_int4_torch,
        _quantize_int4_triton,
    )

    group_size = 32
    torch.manual_seed(12)
    # Per-head magnitudes in [1, 50], peak value stays below ~1.5e3 << fp16 max.
    base = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.float32)
    head_scales = torch.logspace(0, math.log10(50.0), 8, device="cuda", dtype=torch.float32)
    x = (base * head_scales.view(1, 8, 1, 1)).to(torch.bfloat16)
    assert x.float().abs().max().item() < 6e4, "test setup exceeds fp16 range"

    pt, st = _quantize_int4_torch(x, group_size=group_size)
    pk, sk = _quantize_int4_triton(x, group_size=group_size)
    assert torch.isfinite(st).all() and torch.isfinite(sk).all()
    scale_diff = (st.float() - sk.float()).abs().max().item()
    scale_ref = st.float().abs().mean().item()
    assert scale_diff < scale_ref * 1e-3, (
        f"scale mismatch triton vs torch max_abs={scale_diff} vs mean_scale={scale_ref}"
    )

    d_head = int(sk.shape[-1] * group_size)
    x_k = _dequantize_int4_triton(
        pk, sk, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16
    )
    x_t = _dequantize_int4_torch(
        pt, st, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16
    )
    assert torch.isfinite(x_k).all(), "triton dequant non-finite within fp16 range"
    assert torch.isfinite(x_t).all(), "torch dequant non-finite within fp16 range"
    diff = (x_k.float() - x_t.float()).abs().max().item()
    scale_max = sk.float().abs().max().item()
    assert diff < scale_max, (
        f"triton vs torch dequant diff max_abs={diff} vs max_scale={scale_max}"
    )


@cuda
@pytest.mark.xfail(
    reason=(
        "Known sm_70 Triton bug: dequant writes into an fp16 intermediate buffer, "
        "so |lo*scale| > 65504 overflows to inf. Use KV_INT4_TRITON=0 or the sm<8 "
        "autodetect (both routing through torch kernels) to avoid it."
    ),
    strict=True,
)
def test_sm70_triton_overflows_on_fp16_out_of_range() -> None:
    """Pinned repro: Triton dequant produces non-finite values when group max|x| > fp16 max."""
    from kernels.llama3.quantize_sm70 import (
        _dequantize_int4_triton,
        _quantize_int4_triton,
    )

    group_size = 32
    torch.manual_seed(13)
    # Push magnitudes past fp16 max (65504); bf16 input handles it fine.
    x = (torch.randn(1, 4, 32, 128, device="cuda", dtype=torch.float32) * 3e4).to(
        torch.bfloat16
    )
    assert x.float().abs().max().item() > 6.5e4, "test setup should exceed fp16 max"
    pk, sk = _quantize_int4_triton(x, group_size=group_size)
    d_head = int(sk.shape[-1] * group_size)
    x_k = _dequantize_int4_triton(
        pk, sk, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16
    )
    assert torch.isfinite(x_k).all()
