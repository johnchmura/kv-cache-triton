"""INT4 quantization helpers."""

import pytest
import torch

from kernels.gpt2.quantize import dequantize_int4, quantize_int4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_quantize_roundtrip_shape_and_error():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 64, dtype=torch.float16)
    packed, scales = quantize_int4(x, group_size=32)
    x_hat = dequantize_int4(packed, scales, group_size=32, d_head=x.shape[-1])

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
    mae = (x.float() - x_hat.float()).abs().mean().item()
    assert mae < 0.2


def test_quantize_zero_tensor_is_stable():
    x = torch.zeros(1, 2, 4, 64, dtype=torch.float16)
    packed, scales = quantize_int4(x, group_size=32)
    x_hat = dequantize_int4(packed, scales, group_size=32, d_head=x.shape[-1])

    assert torch.allclose(x_hat, x, rtol=0.0, atol=0.0)


def test_quantize_non_power_of_two_width():
    torch.manual_seed(1)
    x = torch.randn(2, 2, 3, 70, dtype=torch.float32)
    packed, scales = quantize_int4(x, group_size=32)
    x_hat = dequantize_int4(packed, scales, group_size=32, d_head=70)

    assert x_hat.shape[-1] == 70
    assert torch.isfinite(x_hat).all()


@cuda
def test_quantize_cuda_matches_cpu_reference_path():
    torch.manual_seed(7)
    x_cuda = torch.randn(2, 3, 5, 64, device="cuda", dtype=torch.float16)

    packed_cuda, scales_cuda = quantize_int4(x_cuda, group_size=32)
    x_hat_cuda = dequantize_int4(packed_cuda, scales_cuda, group_size=32, d_head=64)

    x_cpu = x_cuda.cpu()
    packed_cpu, scales_cpu = quantize_int4(x_cpu, group_size=32)
    x_hat_cpu = dequantize_int4(packed_cpu, scales_cpu, group_size=32, d_head=64)

    packed_diff = (packed_cuda.cpu() != packed_cpu).sum().item()
    assert packed_diff <= 1
    assert torch.equal(scales_cuda.cpu(), scales_cpu)
    delta = (x_hat_cuda.cpu() - x_hat_cpu).abs()
    assert delta.max().item() <= 0.5
    assert delta.mean().item() < 5e-4
