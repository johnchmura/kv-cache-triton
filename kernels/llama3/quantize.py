"""Standalone INT4 groupwise quantize/dequantize for Llama 3.1 KV cache.

Stores the last dim as symmetric INT4 with a per-group scale. Two nibbles are
packed into a single ``uint8`` along the last dim. Scales preserve the input
float dtype (typically bf16 for Llama 3).

Shapes:
    x:      [..., d]       (float: bf16/fp16/fp32)
    packed: [..., d//2]    (uint8)
    scales: [..., d//G]    (same float dtype as x)

The range is symmetric: quantized integers live in ``[-8, 7]`` and are stored
with a ``+8`` bias so they fit an unsigned nibble. The scale is
``max(|x|) / 7`` per group.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_int4_kernel(
    x_ptr,
    packed_ptr,
    scales_ptr,
    stride_xm,
    stride_xd,
    stride_pm,
    stride_pd,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_offs = tl.arange(0, BLOCK_GROUP)
    cols = pid_g * GROUP_SIZE + group_offs
    x_ptrs = x_ptr + pid_m * stride_xm + cols * stride_xd
    x = tl.load(x_ptrs, mask=group_offs < GROUP_SIZE, other=0.0).to(tl.float32)

    max_abs = tl.max(tl.abs(x), axis=0)
    scale = tl.where(max_abs > 0.0, max_abs / 7.0, 1.0)

    pack_per_group = GROUP_SIZE // 2
    pack_offs = tl.arange(0, BLOCK_GROUP // 2)
    valid_pack = pack_offs < pack_per_group
    cols_lo = pid_g * GROUP_SIZE + pack_offs * 2
    cols_hi = cols_lo + 1
    x_lo = tl.load(x_ptr + pid_m * stride_xm + cols_lo * stride_xd, mask=valid_pack, other=0.0).to(tl.float32)
    x_hi = tl.load(x_ptr + pid_m * stride_xm + cols_hi * stride_xd, mask=valid_pack, other=0.0).to(tl.float32)

    q_lo = x_lo / scale
    q_hi = x_hi / scale
    q_lo = tl.where(q_lo >= 0.0, tl.floor(q_lo + 0.5), tl.ceil(q_lo - 0.5))
    q_hi = tl.where(q_hi >= 0.0, tl.floor(q_hi + 0.5), tl.ceil(q_hi - 0.5))
    q_lo = tl.maximum(-8.0, tl.minimum(7.0, q_lo)).to(tl.int32) + 8
    q_hi = tl.maximum(-8.0, tl.minimum(7.0, q_hi)).to(tl.int32) + 8
    q_lo = q_lo.to(tl.uint8)
    q_hi = q_hi.to(tl.uint8)
    packed = q_lo | (q_hi << 4)

    packed_cols = pid_g * pack_per_group + pack_offs
    packed_ptrs = packed_ptr + pid_m * stride_pm + packed_cols * stride_pd
    tl.store(packed_ptrs, packed, mask=valid_pack)

    scale_ptr = scales_ptr + pid_m * stride_sm + pid_g * stride_sg
    tl.store(scale_ptr, scale.to(scales_ptr.dtype.element_ty))


@triton.jit
def _dequantize_int4_kernel(
    packed_ptr,
    scales_ptr,
    out_ptr,
    stride_pm,
    stride_pd,
    stride_sm,
    stride_sg,
    stride_om,
    stride_od,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    pack_per_group = GROUP_SIZE // 2
    pack_offs = tl.arange(0, BLOCK_GROUP // 2)
    valid_pack = pack_offs < pack_per_group
    packed_cols = pid_g * pack_per_group + pack_offs

    p_ptrs = packed_ptr + pid_m * stride_pm + packed_cols * stride_pd
    p = tl.load(p_ptrs, mask=valid_pack, other=0).to(tl.int32)

    lo = (p & 0x0F).to(tl.float32) - 8.0
    hi = ((p >> 4) & 0x0F).to(tl.float32) - 8.0

    scale_ptr = scales_ptr + pid_m * stride_sm + pid_g * stride_sg
    scale = tl.load(scale_ptr).to(tl.float32)
    lo = lo * scale
    hi = hi * scale

    out_cols_lo = pid_g * GROUP_SIZE + pack_offs * 2
    out_cols_hi = out_cols_lo + 1
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_m * stride_om + out_cols_lo * stride_od, lo.to(out_dtype), mask=valid_pack)
    tl.store(out_ptr + pid_m * stride_om + out_cols_hi * stride_od, hi.to(out_dtype), mask=valid_pack)


_SUPPORTED_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def _validate_group_size(group_size: int) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")


def _quantize_int4_torch(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1]
    num_groups = math.ceil(d / group_size)
    padded_d = num_groups * group_size
    pad = padded_d - d

    work = x.float()
    if pad > 0:
        work = F.pad(work, (0, pad))

    grouped = work.view(*work.shape[:-1], num_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    q = torch.round(grouped / scales).clamp(-8, 7).to(torch.int16) + 8
    q = q.to(torch.uint8)

    if group_size % 2 != 0:
        q = F.pad(q, (0, 1))

    lo = q[..., 0::2]
    hi = q[..., 1::2] << 4
    packed = (lo | hi).contiguous().view(*x.shape[:-1], -1)
    return packed, scales.squeeze(-1).to(x.dtype)


def _quantize_int4_triton(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    leading = x.shape[:-1]
    d = x.shape[-1]
    m = math.prod(leading) if len(leading) > 0 else 1
    num_groups = math.ceil(d / group_size)
    padded_d = num_groups * group_size
    pad = padded_d - d

    work = x
    if pad > 0:
        work = F.pad(work, (0, pad))
    work_2d = work.contiguous().view(m, padded_d)

    packed_w = num_groups * (group_size // 2)
    packed_2d = torch.empty((m, packed_w), device=x.device, dtype=torch.uint8)
    scales_2d = torch.empty((m, num_groups), device=x.device, dtype=x.dtype)

    block_group = max(16, triton.next_power_of_2(group_size))
    grid = (m, num_groups)
    _quantize_int4_kernel[grid](
        work_2d,
        packed_2d,
        scales_2d,
        work_2d.stride(0),
        work_2d.stride(1),
        packed_2d.stride(0),
        packed_2d.stride(1),
        scales_2d.stride(0),
        scales_2d.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_GROUP=block_group,
    )

    packed = packed_2d.view(*leading, packed_w)
    scales = scales_2d.view(*leading, num_groups)
    return packed, scales


def quantize_int4(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim to symmetric INT4 with per-group scales.

    Returns ``(packed_uint8, scales)``. ``packed_uint8`` has the same leading
    shape as ``x`` with its last dim halved. ``scales`` preserves the dtype of
    ``x`` and has ``ceil(d / group_size)`` entries per row.
    """
    _validate_group_size(group_size)
    if x.dtype not in _SUPPORTED_FLOATS:
        raise ValueError("quantize_int4 expects bf16/fp16/fp32 input")
    if x.dim() < 1:
        raise ValueError("quantize_int4 expects at least 1D input")

    d = x.shape[-1]
    if d == 0:
        return _quantize_int4_torch(x, group_size)
    if not x.is_cuda or group_size % 2 != 0:
        return _quantize_int4_torch(x, group_size)
    return _quantize_int4_triton(x, group_size)


def _dequantize_int4_torch(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    d_head: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    num_groups = scales.shape[-1]
    unpacked = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        device=packed.device,
        dtype=torch.float32,
    )
    unpacked[..., 0::2] = (packed & 0x0F).to(torch.float32) - 8.0
    unpacked[..., 1::2] = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0

    grouped = unpacked.view(*unpacked.shape[:-1], num_groups, -1)
    if grouped.shape[-1] < group_size:
        raise ValueError("packed has insufficient width for the provided group_size")
    grouped = grouped[..., :group_size]
    out = grouped * scales.float().unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], -1)
    return out[..., :d_head].to(out_dtype)


def _dequantize_int4_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    d_head: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    leading = packed.shape[:-1]
    m = math.prod(leading) if len(leading) > 0 else 1
    num_groups = scales.shape[-1]
    padded_d = num_groups * group_size

    packed_2d = packed.contiguous().view(m, packed.shape[-1])
    scales_2d = scales.contiguous().view(m, num_groups)
    out_2d = torch.empty((m, padded_d), device=packed.device, dtype=out_dtype)

    block_group = max(16, triton.next_power_of_2(group_size))
    grid = (m, num_groups)
    _dequantize_int4_kernel[grid](
        packed_2d,
        scales_2d,
        out_2d,
        packed_2d.stride(0),
        packed_2d.stride(1),
        scales_2d.stride(0),
        scales_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_GROUP=block_group,
    )

    out = out_2d.view(*leading, padded_d)
    return out[..., :d_head]


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    d_head: int | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize a packed INT4 tensor back to ``out_dtype`` (default: scales dtype)."""
    _validate_group_size(group_size)
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")
    if scales.dtype not in _SUPPORTED_FLOATS:
        raise ValueError("scales must be bf16/fp16/fp32")
    if packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError("packed/scales leading dimensions must match")
    if packed.device != scales.device:
        raise ValueError("packed and scales must be on the same device")

    if out_dtype is None:
        out_dtype = scales.dtype
    if out_dtype not in _SUPPORTED_FLOATS:
        raise ValueError("out_dtype must be bf16/fp16/fp32")

    num_groups = scales.shape[-1]
    target_d = num_groups * group_size if d_head is None else int(d_head)
    if target_d <= 0:
        raise ValueError("d_head must be > 0 when provided")

    if not packed.is_cuda or group_size % 2 != 0 or num_groups == 0:
        return _dequantize_int4_torch(
            packed, scales, group_size=group_size, d_head=target_d, out_dtype=out_dtype
        )
    return _dequantize_int4_triton(
        packed, scales, group_size=group_size, d_head=target_d, out_dtype=out_dtype
    )
