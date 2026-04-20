"""INT4 groupwise quantize/dequantize tests for the Llama 3 kernel path."""

from __future__ import annotations

import pytest
import torch

from kernels.llama3.quantize import (
    _dequantize_int4_torch,
    _quantize_int4_torch,
    dequantize_int4,
    quantize_int4,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_round_trip_torch_bf16(group_size: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, 128, dtype=torch.bfloat16)
    packed, scales = _quantize_int4_torch(x, group_size=group_size)
    assert packed.shape[-1] == x.shape[-1] // 2
    assert scales.shape[-1] == x.shape[-1] // group_size
    assert scales.dtype == torch.bfloat16

    x_hat = _dequantize_int4_torch(
        packed, scales, group_size=group_size, d_head=x.shape[-1], out_dtype=torch.bfloat16
    )
    err = (x.float() - x_hat.float()).abs().mean().item()
    scale_mean = scales.float().abs().mean().item()
    assert err < scale_mean, f"mean abs err {err} >= mean scale {scale_mean}"


@cuda
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_triton_matches_torch_bf16(group_size: int) -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, 128, device="cuda", dtype=torch.bfloat16)

    packed_t, scales_t = _quantize_int4_torch(x, group_size=group_size)
    packed_k, scales_k = quantize_int4(x, group_size=group_size)
    assert scales_t.dtype == scales_k.dtype
    assert torch.allclose(scales_t.float(), scales_k.float(), rtol=1e-2, atol=1e-2)

    x_hat_t = _dequantize_int4_torch(
        packed_t, scales_t, group_size=group_size, d_head=x.shape[-1], out_dtype=torch.bfloat16
    )
    x_hat_k = dequantize_int4(packed_k, scales_k, group_size=group_size, d_head=x.shape[-1])
    tri_vs_torch = (x_hat_t.float() - x_hat_k.float()).abs().mean().item()
    scale_mean = scales_k.float().abs().mean().item()
    assert tri_vs_torch < scale_mean * 0.1, (
        f"triton vs torch mean abs diff {tri_vs_torch} too large vs scale {scale_mean}"
    )


@cuda
def test_scales_dtype_tracks_input() -> None:
    x = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.bfloat16)
    _, scales = quantize_int4(x, group_size=32)
    assert scales.dtype == torch.bfloat16

    y = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float16)
    _, scales_fp16 = quantize_int4(y, group_size=32)
    assert scales_fp16.dtype == torch.float16


@cuda
def test_dequantize_out_dtype() -> None:
    x = torch.randn(1, 2, 8, 128, device="cuda", dtype=torch.bfloat16)
    packed, scales = quantize_int4(x, group_size=32)
    out_bf16 = dequantize_int4(packed, scales, group_size=32, d_head=128)
    out_fp16 = dequantize_int4(packed, scales, group_size=32, d_head=128, out_dtype=torch.float16)
    assert out_bf16.dtype == torch.bfloat16
    assert out_fp16.dtype == torch.float16
    assert torch.allclose(out_bf16.float(), out_fp16.float(), rtol=5e-3, atol=5e-3)


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        quantize_int4(torch.randn(4, 128), group_size=0)
    with pytest.raises(ValueError):
        quantize_int4(torch.randint(0, 5, (4, 128), dtype=torch.int32), group_size=32)
