"""Fused attention over INT4-packed K/V cache."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd_quant_kernel(
    q_ptr,
    k_packed_ptr,
    k_scales_ptr,
    v_packed_ptr,
    v_scales_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_sq,
    stride_q_d,
    stride_kp_b,
    stride_kp_h,
    stride_kp_sk,
    stride_kp_dpack,
    stride_ks_b,
    stride_ks_h,
    stride_ks_sk,
    stride_ks_g,
    stride_vp_b,
    stride_vp_h,
    stride_vp_sk,
    stride_vp_dpack,
    stride_vs_b,
    stride_vs_h,
    stride_vs_sk,
    stride_vs_g,
    stride_o_b,
    stride_o_h,
    stride_o_sq,
    stride_o_d,
    seq_len_q,
    seq_len_k,
    d_head,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    kp_base = k_packed_ptr + pid_b * stride_kp_b + pid_h * stride_kp_h
    ks_base = k_scales_ptr + pid_b * stride_ks_b + pid_h * stride_ks_h
    vp_base = v_packed_ptr + pid_b * stride_vp_b + pid_h * stride_vp_h
    vs_base = v_scales_ptr + pid_b * stride_vs_b + pid_h * stride_vs_h
    o_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h

    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + q_offs[:, None] * stride_q_sq + d_offs[None, :] * stride_q_d
    q_mask = (q_offs[:, None] < seq_len_q) & (d_offs[None, :] < d_head)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_Q], float("-1e30"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    causal_offset = seq_len_k - seq_len_q
    if IS_CAUSAL:
        k_end = causal_offset + (pid_q + 1) * BLOCK_Q
        if k_end > seq_len_k:
            k_end = seq_len_k
    else:
        k_end = seq_len_k

    d_pack = d_offs // 2
    d_hi = (d_offs % 2) == 1
    d_group = d_offs // GROUP_SIZE
    d_mask = d_offs < d_head

    for k_start in range(0, k_end, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        kv_mask = k_offs[:, None] < seq_len_k
        mask_full = kv_mask & d_mask[None, :]

        kp_ptrs = kp_base + k_offs[:, None] * stride_kp_sk + d_pack[None, :] * stride_kp_dpack
        kp = tl.load(kp_ptrs, mask=mask_full, other=0)
        kp_i32 = kp.to(tl.int32)
        nibble_low = kp_i32 & 0x0F
        nibble_high = (kp_i32 >> 4) & 0x0F
        k_nibble = tl.where(d_hi[None, :], nibble_high, nibble_low)

        ks_ptrs = ks_base + k_offs[:, None] * stride_ks_sk + d_group[None, :] * stride_ks_g
        k_scales = tl.load(ks_ptrs, mask=mask_full, other=1.0).to(tl.float32)
        k = (k_nibble.to(tl.float32) - 8.0) * k_scales

        s = tl.dot(q, tl.trans(k)) * scale
        valid = (q_offs[:, None] < seq_len_q) & (k_offs[None, :] < seq_len_k)
        if IS_CAUSAL:
            valid = valid & ((q_offs[:, None] + causal_offset) >= k_offs[None, :])
        s = tl.where(valid, s, float("-1e30"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None]

        vp_ptrs = vp_base + k_offs[:, None] * stride_vp_sk + d_pack[None, :] * stride_vp_dpack
        vp = tl.load(vp_ptrs, mask=mask_full, other=0)
        vp_i32 = vp.to(tl.int32)
        v_nibble_low = vp_i32 & 0x0F
        v_nibble_high = (vp_i32 >> 4) & 0x0F
        v_nibble = tl.where(d_hi[None, :], v_nibble_high, v_nibble_low)

        vs_ptrs = vs_base + k_offs[:, None] * stride_vs_sk + d_group[None, :] * stride_vs_g
        v_scales = tl.load(vs_ptrs, mask=mask_full, other=1.0).to(tl.float32)
        v = (v_nibble.to(tl.float32) - 8.0) * v_scales

        o_i += tl.dot(p.to(tl.float32), v)
        m_i = m_new

    o_i = o_i / l_i[:, None]

    o_ptrs = o_base + q_offs[:, None] * stride_o_sq + d_offs[None, :] * stride_o_d
    o_mask = (q_offs[:, None] < seq_len_q) & (d_offs[None, :] < d_head)
    tl.store(o_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=o_mask)


def attention_forward_quant(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    *,
    group_size: int = 32,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention using packed INT4 K/V cache."""
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if not q.is_cuda:
        raise ValueError("attention_forward_quant expects CUDA tensors")
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError("q must be float16 or float32")
    if k_packed.dtype != torch.uint8 or v_packed.dtype != torch.uint8:
        raise ValueError("k_packed and v_packed must be uint8")
    if k_scales.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("k_scales must be floating point")
    if v_scales.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("v_scales must be floating point")

    if not all(t.is_cuda for t in (k_packed, k_scales, v_packed, v_scales)):
        raise ValueError("attention_forward_quant expects CUDA tensors")

    if k_packed.shape != v_packed.shape:
        raise ValueError("k_packed and v_packed must have the same shape")
    if k_scales.shape != v_scales.shape:
        raise ValueError("k_scales and v_scales must have the same shape")
    if k_packed.shape[:-1] != k_scales.shape[:-1]:
        raise ValueError("packed/scales leading dimensions must match")

    b, h, sq, d = q.shape
    if k_packed.shape[:2] != (b, h):
        raise ValueError("batch/head dimensions must match q")
    if k_scales.shape[:2] != (b, h):
        raise ValueError("batch/head dimensions must match q")

    sk = k_packed.shape[2]
    min_packed = math.ceil(d / 2)
    min_groups = math.ceil(d / group_size)
    if k_packed.shape[-1] < min_packed:
        raise ValueError("packed last dimension is too small for q d_head")
    if k_scales.shape[-1] < min_groups:
        raise ValueError("scales last dimension is too small for q d_head/group_size")

    q = q.contiguous()
    k_packed = k_packed.contiguous()
    v_packed = v_packed.contiguous()
    k_scales = k_scales.contiguous()
    v_scales = v_scales.contiguous()

    out = torch.empty_like(q)
    scale = d ** -0.5

    block_q = max(16, min(64, triton.next_power_of_2(sq)))
    block_k = max(16, min(64, triton.next_power_of_2(sk)))
    block_d = max(16, triton.next_power_of_2(d))

    num_q_blocks = triton.cdiv(sq, block_q)
    grid = (b, h, num_q_blocks)
    num_warps = 4 if block_q * block_k <= 4096 else 8

    _attention_fwd_quant_kernel[grid](
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_packed.stride(0),
        k_packed.stride(1),
        k_packed.stride(2),
        k_packed.stride(3),
        k_scales.stride(0),
        k_scales.stride(1),
        k_scales.stride(2),
        k_scales.stride(3),
        v_packed.stride(0),
        v_packed.stride(1),
        v_packed.stride(2),
        v_packed.stride(3),
        v_scales.stride(0),
        v_scales.stride(1),
        v_scales.stride(2),
        v_scales.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        sq,
        sk,
        d,
        scale,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        GROUP_SIZE=group_size,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
