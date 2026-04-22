"""Volta-safe (sm_70) variant of the fused GQA INT4 attention kernel.

The Triton kernel body is identical to ``kernels/llama3/attention_quant.py``
(math already runs in ``tl.float32``). The only reason BF16 PTX (``.bf16``,
``cvt.bf16.f32``, ``cvt.f32.bf16``) leaks in on the original path is the
implicit load/store conversions Triton emits when the pointer dtype is bf16.

The Python wrapper keeps the Triton boundary clean:

- Q is cast to fp16 if the caller passed bf16 (kernel loads Q via fp16 PTX).
- The output buffer is fp32 so the final ``o_i / l_i`` can be stored without
  an fp16 overflow or precision loss, then cast to the caller dtype on the
  torch side. An earlier fp16 output buffer caused real-Llama-magnitude
  attention outputs to saturate to +/-inf during prefill, driving PPL to
  ~2e6 on the benchmark.
- Scales stay in fp32 end-to-end. The sm_70 cache stores fp32 scales, and
  the kernel reloads them as fp32 via ``tl.load(...).to(tl.float32)``, so
  there is no reason to round-trip through fp16 at the boundary.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd_quant_gqa_kernel_sm70(
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
    GQA_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    pid_kv = pid_h // GQA_RATIO

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    kp_base = k_packed_ptr + pid_b * stride_kp_b + pid_kv * stride_kp_h
    ks_base = k_scales_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    vp_base = v_packed_ptr + pid_b * stride_vp_b + pid_kv * stride_vp_h
    vs_base = v_scales_ptr + pid_b * stride_vs_b + pid_kv * stride_vs_h
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

    # Guard against all-masked rows where every score was -1e30 and l_i stayed 0.
    # Without this, the final divide produces nan/inf and poisons downstream layers.
    l_safe = tl.maximum(l_i, 1e-30)
    o_i = o_i / l_safe[:, None]

    o_ptrs = o_base + q_offs[:, None] * stride_o_sq + d_offs[None, :] * stride_o_d
    o_mask = (q_offs[:, None] < seq_len_q) & (d_offs[None, :] < d_head)
    tl.store(o_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=o_mask)


_Q_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SCALE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _to_sm70_q_dtype(t: torch.Tensor) -> torch.Tensor:
    """Cast Q to fp16 when caller is bf16 so Triton never loads bf16 on sm_70."""
    if t.dtype == torch.bfloat16:
        return t.to(torch.float16)
    return t


def _to_sm70_scale_dtype(t: torch.Tensor) -> torch.Tensor:
    """Scales cross the Triton boundary as fp32 on sm_70.

    Historical versions downcast to fp16 here (to match the Q dtype routing),
    but fp16 narrow-range underflow silently zeroed tiny groups, so the cache
    has stored scales in fp32 for a while. The kernel reloads scales as fp32
    anyway; widening bf16/fp16 inputs to fp32 is a no-op for precision and
    avoids a round-trip that could underflow.
    """
    if t.dtype == torch.float32:
        return t
    return t.to(torch.float32)


def attention_forward_quant_gqa_sm70(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    *,
    n_q_heads: int,
    n_kv_heads: int,
    group_size: int = 32,
    is_causal: bool = False,
) -> torch.Tensor:
    """Volta-safe fused GQA attention over packed INT4 K/V.

    Same contract as :func:`kernels.llama3.attention_quant.attention_forward_quant_gqa`.
    All bf16 float tensors are cast to fp16 before launch and the output is
    cast back to ``q.dtype``. No tensor with bf16 element dtype ever reaches
    the Triton kernel.
    """
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if n_q_heads <= 0 or n_kv_heads <= 0:
        raise ValueError("n_q_heads and n_kv_heads must be > 0")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")

    if not q.is_cuda:
        raise ValueError("attention_forward_quant_gqa_sm70 expects CUDA tensors")
    if not all(t.is_cuda for t in (k_packed, k_scales, v_packed, v_scales)):
        raise ValueError("attention_forward_quant_gqa_sm70 expects CUDA tensors")

    if q.dtype not in _Q_DTYPES:
        raise ValueError("q must be bf16/fp16/fp32")
    if k_packed.dtype != torch.uint8 or v_packed.dtype != torch.uint8:
        raise ValueError("k_packed and v_packed must be uint8")
    if k_scales.dtype not in _SCALE_DTYPES or v_scales.dtype not in _SCALE_DTYPES:
        raise ValueError("scales must be bf16/fp16/fp32")

    if k_packed.shape != v_packed.shape:
        raise ValueError("k_packed and v_packed must have the same shape")
    if k_scales.shape != v_scales.shape:
        raise ValueError("k_scales and v_scales must have the same shape")
    if k_packed.shape[:-1] != k_scales.shape[:-1]:
        raise ValueError("packed/scales leading dimensions must match")

    if q.dim() != 4:
        raise ValueError("q must be 4D: [B, n_q_heads, sq, d_head]")
    b, hq, sq, d = q.shape
    if hq != n_q_heads:
        raise ValueError(f"q head dim {hq} != n_q_heads {n_q_heads}")
    if k_packed.shape[0] != b or k_packed.shape[1] != n_kv_heads:
        raise ValueError("k/v leading dims must be [B, n_kv_heads, ...]")

    sk = k_packed.shape[2]
    min_packed = math.ceil(d / 2)
    min_groups = math.ceil(d / group_size)
    if k_packed.shape[-1] < min_packed:
        raise ValueError("packed last dimension is too small for q d_head")
    if k_scales.shape[-1] < min_groups:
        raise ValueError("scales last dimension is too small for q d_head/group_size")

    caller_dtype = q.dtype

    q_k = _to_sm70_q_dtype(q).contiguous()
    k_scales_k = _to_sm70_scale_dtype(k_scales).contiguous()
    v_scales_k = _to_sm70_scale_dtype(v_scales).contiguous()
    k_packed = k_packed.contiguous()
    v_packed = v_packed.contiguous()

    # Output buffer is fp32 so o_i / l_i has room to store values above the
    # fp16 range without saturating. Caller-dtype cast happens on the torch
    # side below. Emits cvt.f32.f16 / cvt.f16.f32 only - sm_70 legal.
    out = torch.empty(q_k.shape, device=q_k.device, dtype=torch.float32)
    scale = d ** -0.5
    gqa_ratio = n_q_heads // n_kv_heads

    block_q = max(16, min(64, triton.next_power_of_2(sq)))
    block_k = max(16, min(64, triton.next_power_of_2(sk)))
    block_d = max(16, triton.next_power_of_2(d))

    num_q_blocks = triton.cdiv(sq, block_q)
    grid = (b, n_q_heads, num_q_blocks)
    num_warps = 4 if block_q * block_k <= 4096 else 8

    _attention_fwd_quant_gqa_kernel_sm70[grid](
        q_k,
        k_packed,
        k_scales_k,
        v_packed,
        v_scales_k,
        out,
        q_k.stride(0),
        q_k.stride(1),
        q_k.stride(2),
        q_k.stride(3),
        k_packed.stride(0),
        k_packed.stride(1),
        k_packed.stride(2),
        k_packed.stride(3),
        k_scales_k.stride(0),
        k_scales_k.stride(1),
        k_scales_k.stride(2),
        k_scales_k.stride(3),
        v_packed.stride(0),
        v_packed.stride(1),
        v_packed.stride(2),
        v_packed.stride(3),
        v_scales_k.stride(0),
        v_scales_k.stride(1),
        v_scales_k.stride(2),
        v_scales_k.stride(3),
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
        GQA_RATIO=gqa_ratio,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        # num_stages=1 on Volta: sm_70 lacks fp32 tensor cores, so tl.dot runs
        # on CUDA cores and multi-stage pipelining has produced inconsistent
        # results on this kernel shape. Single stage is slower but correct.
        num_stages=1,
    )
    if out.dtype != caller_dtype:
        out = out.to(caller_dtype)
    return out
