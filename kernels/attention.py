"""Triton attention kernels.

Two specialised kernels:
  * _decode_kernel  – single-query (q_len == 1), element-wise, no tl.dot overhead.
  * _prefill_kernel – arbitrary q_len, flash-attention tiling with online softmax.

The public ``attention_forward`` dispatches automatically based on q_len.
"""

import torch
import triton
import triton.language as tl
from typing import Optional



# Decode kernel (INT4 Ready): Q [B, H, 1, D], K/V [B, H, S, D//2] (uint8) + scales


@triton.jit
def _decode_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    stride_q_b, stride_q_h, stride_q_1, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_ksc_b, stride_ksc_h, stride_ksc_s,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_vsc_b, stride_vsc_h, stride_vsc_s,
    stride_o_b, stride_o_h, stride_o_1, stride_o_d,
    seq_len,
    d_head, # This is the full output D natively (e.g. 64)
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,  # This will be full D
    IS_FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_packed_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_packed_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    ksc_base = k_scale_ptr + pid_b * stride_ksc_b + pid_h * stride_ksc_h
    vsc_base = v_scale_ptr + pid_b * stride_vsc_b + pid_h * stride_vsc_h
    o_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_head
    q = tl.load(q_base + offs_d * stride_q_d, mask=mask_d, other=0.0)
    q = q.to(tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < seq_len
    
    # We load D//2 packed uint8 elements
    offs_d_packed = tl.arange(0, BLOCK_D // 2)
    mask_d_packed = offs_d_packed < (d_head // 2)
    
    k_ptrs = k_base + offs_s[:, None] * stride_k_s + offs_d_packed[None, :] * stride_k_d
    k_mask = mask_s[:, None] & mask_d_packed[None, :]
    k_packed = tl.load(k_ptrs, mask=k_mask, other=0) # [BLOCK_S, BLOCK_D//2] uint8
    
    # Unpack into [BLOCK_S, BLOCK_D//2] int8 ranges
    k_low = (k_packed & 0x0F).to(tl.int8) - 8
    k_high = ((k_packed >> 4) & 0x0F).to(tl.int8) - 8
    
    # We want to interleave k_low and k_high to form [BLOCK_S, BLOCK_D]
    # In Triton arrays interact element-wise. We can build the full vector by index math or reshape.
    # PyTorch stack + view packs ADJACENT pairs. byte 0 packs 0 and 1. byte 1 packs 2 and 3.
    # Therefore, k_low matches q[0, 2, 4...] and k_high matches q[1, 3, 5...]
    
    q_low = tl.load(q_base + (2 * offs_d_packed) * stride_q_d, mask=mask_d_packed, other=0.0).to(tl.float32)
    q_high = tl.load(q_base + (2 * offs_d_packed + 1) * stride_q_d, mask=mask_d_packed, other=0.0).to(tl.float32)
    
    k_scales = tl.load(ksc_base + offs_s * stride_ksc_s, mask=mask_s, other=0.0).to(tl.float32)
    
    # Q * K = (Q_low * K_low_fp16 + Q_high * K_high_fp16) -> then sum across D
    k_low_scaled = k_low.to(tl.float32) * k_scales[:, None]
    k_high_scaled = k_high.to(tl.float32) * k_scales[:, None]
    
    logits_low = tl.sum(q_low[None, :] * k_low_scaled, axis=1)
    logits_high = tl.sum(q_high[None, :] * k_high_scaled, axis=1)
    
    logits = (logits_low + logits_high) * scale
    logits = tl.where(mask_s, logits, float("-inf"))

    m = tl.max(logits)
    logits = logits - m
    w = tl.exp(logits)
    denom = tl.sum(w)
    attn = w / denom

    # Value calculation
    v_ptrs = v_base + offs_s[:, None] * stride_v_s + offs_d_packed[None, :] * stride_v_d
    v_packed = tl.load(v_ptrs, mask=k_mask, other=0)
    
    v_low = (v_packed & 0x0F).to(tl.int8) - 8
    v_high = ((v_packed >> 4) & 0x0F).to(tl.int8) - 8
    
    v_scales = tl.load(vsc_base + offs_s * stride_vsc_s, mask=mask_s, other=0.0).to(tl.float32)
    v_low_scaled = v_low.to(tl.float32) * v_scales[:, None]
    v_high_scaled = v_high.to(tl.float32) * v_scales[:, None]
    
    out_low = tl.sum(attn[:, None] * v_low_scaled, axis=0)
    out_high = tl.sum(attn[:, None] * v_high_scaled, axis=0)
    
    if IS_FP16:
        tl.store(o_base + (2 * offs_d_packed) * stride_o_d, out_low.to(tl.float16), mask=mask_d_packed)
        tl.store(o_base + (2 * offs_d_packed + 1) * stride_o_d, out_high.to(tl.float16), mask=mask_d_packed)
    else:
        tl.store(o_base + (2 * offs_d_packed) * stride_o_d, out_low, mask=mask_d_packed)
        tl.store(o_base + (2 * offs_d_packed + 1) * stride_o_d, out_high, mask=mask_d_packed)


# ---------------------------------------------------------------------------
# Prefill kernel: Q [B, H, Sq, D], K/V [B, H, Sk, D], flash-attention tiling
# ---------------------------------------------------------------------------

@triton.jit
def _prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_sq,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_sk,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_sk,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_sq,
    stride_o_d,
    seq_len_q,
    seq_len_k,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    o_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h

    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)

    # Load Q tile: [BLOCK_Q, BLOCK_D]
    q_ptrs = q_base + q_offs[:, None] * stride_q_sq + d_offs[None, :] * stride_q_d
    q_mask = q_offs[:, None] < seq_len_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Online softmax accumulators.
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

    for k_start in range(0, k_end, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        kv_mask = k_offs[:, None] < seq_len_k

        k_ptrs = k_base + k_offs[:, None] * stride_k_sk + d_offs[None, :] * stride_k_d
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

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

        v_ptrs = v_base + k_offs[:, None] * stride_v_sk + d_offs[None, :] * stride_v_d
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        o_i += tl.dot(p.to(tl.float32), v)

        m_i = m_new

    o_i = o_i / l_i[:, None]

    o_ptrs = o_base + q_offs[:, None] * stride_o_sq + d_offs[None, :] * stride_o_d
    o_mask = q_offs[:, None] < seq_len_q
    tl.store(o_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    k_scales: Optional[torch.Tensor] = None,
    v_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention – dispatches to the optimal kernel.

    * q_len == 1 (decode): expects `k` and `v` to be uint8 INT4 packed, with scales.
    * q_len >  1 (prefill): expects `k` and `v` to be standard float16.
    """
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("attention_forward expects CUDA tensors")
    dtype = q.dtype
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("attention_forward supports float16 and float32")

    b, h, sq, d = q.shape
    sk = k.shape[2]
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    scale = d ** -0.5

    if sq == 1:
        # ---- Decode path: unpacks INT4 cached K/V ----
        block_s = triton.next_power_of_2(sk)
        block_d = triton.next_power_of_2(d)
        is_fp16 = q.dtype == torch.float16
        
        if k_scales is not None:
            k_scales = k_scales.contiguous()
        if v_scales is not None:
            v_scales = v_scales.contiguous()

        _decode_kernel[(b, h)](
            q, k, v, k_scales, v_scales, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            sk, d, scale,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
            IS_FP16=is_fp16,
        )
    else:
        # ---- Prefill path: flash-attention tiling ----
        BLOCK_Q = max(16, min(64, triton.next_power_of_2(sq)))
        BLOCK_K = max(16, min(64, triton.next_power_of_2(sk)))
        BLOCK_D = max(16, triton.next_power_of_2(d))
        num_q_blocks = triton.cdiv(sq, BLOCK_Q)
        num_warps = 4 if BLOCK_Q * BLOCK_K <= 4096 else 8

        _prefill_kernel[(b, h, num_q_blocks)](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            sq, sk, scale,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            num_warps=num_warps,
            num_stages=2,
        )

    return out
