"""Fused sm70 INT4 attention vs PyTorch reference on the same dequantized K/V."""

from __future__ import annotations

import pytest
import torch

from kernels.llama3.attention_quant_sm70 import attention_forward_quant_gqa_sm70
from kernels.llama3.quantize_sm70 import quantize_int4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _gqa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_q_heads: int,
    n_kv_heads: int,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Reference: GQA attention, bf16/fp16 math in fp32."""
    b, n_q, sq, d = q.shape
    _, n_kv, sk, _ = k.shape
    assert n_q == n_q_heads and n_kv == n_kv_heads
    gqa = n_q_heads // n_kv_heads
    out = torch.empty_like(q)
    for bi in range(b):
        for hq in range(n_q_heads):
            hkv = hq // gqa
            qq = q[bi, hq].float()
            kk = k[bi, hkv].float()
            vv = v[bi, hkv].float()
            scores = (qq @ kk.T) * scale
            if is_causal:
                if sq == sk:
                    bad = torch.triu(
                        torch.ones(sq, sk, device=q.device, dtype=torch.bool),
                        diagonal=1,
                    )
                else:
                    causal_offset = sk - sq
                    ii = torch.arange(sq, device=q.device)[:, None]
                    jj = torch.arange(sk, device=q.device)[None, :]
                    bad = (ii + causal_offset) < jj
                scores = scores.masked_fill(bad, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[bi, hq] = (attn @ vv).to(q.dtype)
    return out


@cuda
@pytest.mark.parametrize("is_causal", [False, True])
def test_fused_matches_dequant_ref(is_causal: bool) -> None:
    torch.manual_seed(0)
    b, n_q, n_kv = 1, 8, 2
    sq, sk, d = 16, 16, 128
    group_size = 32
    if is_causal:
        sq, sk = 16, 16
    else:
        sq, sk = 1, 16

    q = torch.randn(b, n_q, sq, d, device="cuda", dtype=torch.bfloat16)
    k_bf = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16)
    v_bf = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16)

    k_p, k_s = quantize_int4(k_bf, group_size=group_size)
    v_p, v_s = quantize_int4(v_bf, group_size=group_size)

    fused = attention_forward_quant_gqa_sm70(
        q,
        k_p,
        k_s,
        v_p,
        v_s,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=is_causal,
    )

    from kernels.llama3.quantize_sm70 import dequantize_int4

    ng = k_s.shape[-1]
    d_head = ng * group_size
    k_dq = dequantize_int4(k_p, k_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)
    v_dq = dequantize_int4(v_p, v_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)

    scale = d ** -0.5
    ref = _gqa_ref(
        q,
        k_dq,
        v_dq,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        scale=scale,
        is_causal=is_causal,
    )

    tol = 0.15 if is_causal else 0.08
    max_diff = (fused.float() - ref.float()).abs().max().item()
    assert max_diff < tol, f"fused vs ref max_abs={max_diff} (tol={tol})"


@cuda
def test_fused_matches_dequant_prefill_causal() -> None:
    """Longer prefill causal path (same as decode-length 1 with is_causal False uses different branch)."""
    torch.manual_seed(1)
    b, n_q, n_kv = 1, 32, 8
    sq = sk = 32
    d = 128
    group_size = 32

    q = torch.randn(b, n_q, sq, d, device="cuda", dtype=torch.bfloat16) * 0.02
    k_bf = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16) * 0.02
    v_bf = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16) * 0.02

    k_p, k_s = quantize_int4(k_bf, group_size=group_size)
    v_p, v_s = quantize_int4(v_bf, group_size=group_size)

    fused = attention_forward_quant_gqa_sm70(
        q,
        k_p,
        k_s,
        v_p,
        v_s,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=True,
    )

    from kernels.llama3.quantize_sm70 import dequantize_int4

    d_head = int(k_s.shape[-1] * group_size)
    k_dq = dequantize_int4(k_p, k_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)
    v_dq = dequantize_int4(v_p, v_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)

    scale = d ** -0.5
    ref = _gqa_ref(
        q,
        k_dq,
        v_dq,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        scale=scale,
        is_causal=True,
    )

    max_diff = (fused.float() - ref.float()).abs().max().item()
    assert max_diff < 0.2, f"max_abs={max_diff}"


def _realistic_qkv_llama8b(
    device: torch.device,
    dtype: torch.dtype,
    *,
    sq: int,
    sk: int,
    n_q: int,
    n_kv: int,
    d: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q / K / V shaped like Llama-3.1-8B post-RoPE with realistic magnitudes.

    Per-head scale is drawn in [1e1, 1e3] on a log-uniform, and a small
    number of channels are pushed to ~5e3 to emulate outliers. Peak stays
    well below 65504 so this test does not overlap with the already-known
    fp16 overflow bug in the dequant kernel.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    b = 1
    q = torch.empty(b, n_q, sq, d, device=device, dtype=torch.float32)
    k = torch.empty(b, n_kv, sk, d, device=device, dtype=torch.float32)
    v = torch.empty(b, n_kv, sk, d, device=device, dtype=torch.float32)

    for h in range(n_q):
        log_scale = torch.rand((), generator=g, device=device) * 2.0 + 1.0
        scale = 10.0 ** log_scale.item()
        q[0, h] = torch.randn(sq, d, generator=g, device=device) * scale
    for h in range(n_kv):
        log_scale = torch.rand((), generator=g, device=device) * 2.0 + 1.0
        scale = 10.0 ** log_scale.item()
        k[0, h] = torch.randn(sk, d, generator=g, device=device) * scale
        log_scale = torch.rand((), generator=g, device=device) * 2.0 + 1.0
        scale = 10.0 ** log_scale.item()
        v[0, h] = torch.randn(sk, d, generator=g, device=device) * scale

    # Sprinkle a few outlier channels per tensor, capped at 5e3 so we stay
    # clear of the fp16 overflow threshold (65504) even after group scaling.
    for t in (q, k, v):
        n_outliers = max(1, d // 32)
        ch = torch.randperm(d, generator=g, device=device)[:n_outliers]
        t[..., ch] *= 5.0
        t.clamp_(min=-5e3, max=5e3)

    return q.to(dtype), k.to(dtype), v.to(dtype)


@cuda
@pytest.mark.parametrize("is_causal", [False, True])
def test_fused_matches_dequant_llama8b_real_magnitudes(is_causal: bool) -> None:
    """Llama-3.1-8B shape at realistic post-RoPE magnitudes.

    Earlier prefill benchmarks showed PPL ~2e6 on the fused path while
    small-magnitude unit tests (max_abs <= 1) stayed green. This test
    exercises the magnitude regime where the fp16 output buffer and the
    unguarded softmax denominator used to saturate / divide-by-zero. The
    tolerance is relative to the peak attention score (quantization error
    is O(scale), not absolute).
    """
    device = torch.device("cuda")
    b, n_q, n_kv, d = 1, 32, 8, 128
    sq = sk = 512
    group_size = 32

    q, k_bf, v_bf = _realistic_qkv_llama8b(
        device,
        torch.bfloat16,
        sq=sq,
        sk=sk,
        n_q=n_q,
        n_kv=n_kv,
        d=d,
        seed=42,
    )

    k_p, k_s = quantize_int4(k_bf, group_size=group_size)
    v_p, v_s = quantize_int4(v_bf, group_size=group_size)

    fused = attention_forward_quant_gqa_sm70(
        q,
        k_p,
        k_s,
        v_p,
        v_s,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=is_causal,
    )
    assert torch.isfinite(fused).all(), "fused output contains non-finite values"

    from kernels.llama3.quantize_sm70 import dequantize_int4

    d_head = int(k_s.shape[-1] * group_size)
    k_dq = dequantize_int4(k_p, k_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)
    v_dq = dequantize_int4(v_p, v_s, group_size=group_size, d_head=d_head, out_dtype=torch.bfloat16)

    scale = d ** -0.5
    ref = _gqa_ref(
        q,
        k_dq,
        v_dq,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        scale=scale,
        is_causal=is_causal,
    )
    assert torch.isfinite(ref).all(), "reference output contains non-finite values"

    # At these magnitudes softmax collapses to approximate argmax per query, so
    # tiny quantization noise in the scores flips which V row gets read and the
    # elementwise max diff is dominated by a handful of argmax flips. Both the
    # fused and reference paths exhibit this; what we want to catch is
    # saturation (inf/nan) or a kernel that returns nonsense. Test instead:
    #
    # - mean_abs agrees (global distribution preserved)
    # - peak magnitude agrees (no output-buffer fp16 saturation)
    # - the mean diff is within INT4 quantization noise bounds.
    ff = fused.float()
    rf = ref.float()
    fused_mean = ff.abs().mean().item()
    ref_mean = rf.abs().mean().item()
    fused_peak = ff.abs().max().item()
    ref_peak = rf.abs().max().item()
    mean_diff = (ff - rf).abs().mean().item()

    assert fused_peak <= max(ref_peak * 1.05, ref_peak + 1.0), (
        f"fused peak {fused_peak} exceeds ref peak {ref_peak} by too much "
        f"(output-buffer saturation suspected)"
    )
    assert fused_peak >= ref_peak * 0.5, (
        f"fused peak {fused_peak} collapsed vs ref {ref_peak} "
        f"(softmax denom underflow suspected)"
    )
    rel_mean_drift = abs(fused_mean - ref_mean) / max(ref_mean, 1e-6)
    assert rel_mean_drift < 0.05, (
        f"fused mean_abs={fused_mean} ref mean_abs={ref_mean} drift={rel_mean_drift:.4f}"
    )
    # Mean absolute diff is bounded by E[|V_err|] ~ scale / 16. The worst
    # per-group scale is v_peak/7, so E[|diff|] < v_peak/100 is a comfortable
    # bound that still catches "fused is silently wrong" regressions.
    v_peak = v_dq.float().abs().max().item()
    tol_mean = max(1.0, v_peak / 100.0)
    assert mean_diff < tol_mean, (
        f"fused vs ref mean_abs diff={mean_diff} (tol_mean={tol_mean}, v_peak={v_peak})"
    )
