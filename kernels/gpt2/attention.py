"""Scaled dot-product attention with flash-attention tiling.

Supports arbitrary query lengths: Q [B,H,Sq,D], K/V [B,H,Sk,D].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd_kernel(
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

    # Base pointers for this (batch, head).
    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    o_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h

    # Offsets for the query block processed by this program.
    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)

    # Load Q tile: [BLOCK_Q, BLOCK_D]
    q_ptrs = q_base + q_offs[:, None] * stride_q_sq + d_offs[None, :] * stride_q_d
    q_mask = q_offs[:, None] < seq_len_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Online softmax accumulators (per query row).
    m_i = tl.full([BLOCK_Q], float("-1e30"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # For causal masking the absolute position of q[i] is (sk - sq) + q_offs[i].
    causal_offset = seq_len_k - seq_len_q

    # Upper bound on key positions to visit (skip fully masked blocks for causal).
    if IS_CAUSAL:
        k_end = causal_offset + (pid_q + 1) * BLOCK_Q
        if k_end > seq_len_k:
            k_end = seq_len_k
    else:
        k_end = seq_len_k

    for k_start in range(0, k_end, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        kv_mask = k_offs[:, None] < seq_len_k

        # Load K tile: [BLOCK_K, BLOCK_D]
        k_ptrs = k_base + k_offs[:, None] * stride_k_sk + d_offs[None, :] * stride_k_d
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_Q, BLOCK_K]
        s = tl.dot(q, tl.trans(k)) * scale

        # Validity mask (out-of-bounds → -1e30).
        valid = (q_offs[:, None] < seq_len_q) & (k_offs[None, :] < seq_len_k)
        if IS_CAUSAL:
            valid = valid & ((q_offs[:, None] + causal_offset) >= k_offs[None, :])
        s = tl.where(valid, s, float("-1e30"))

        # Online softmax update.
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None]

        # Load V tile and accumulate: P @ V
        v_ptrs = v_base + k_offs[:, None] * stride_v_sk + d_offs[None, :] * stride_v_d
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        o_i += tl.dot(p.to(tl.float32), v)

        m_i = m_new

    o_i = o_i / l_i[:, None]

    # Store output tile: [BLOCK_Q, BLOCK_D]
    o_ptrs = o_base + q_offs[:, None] * stride_o_sq + d_offs[None, :] * stride_o_d
    o_mask = q_offs[:, None] < seq_len_q
    tl.store(o_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=o_mask)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention for arbitrary query and key lengths.

    Shapes
    ------
    q : [B, H, Sq, D]
    k : [B, H, Sk, D]
    v : [B, H, Sk, D]
    Returns : [B, H, Sq, D]

    When ``is_causal=True`` each query position attends only to key positions
    at or before its absolute position.
    """
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("attention_forward expects CUDA tensors")
    dtype = q.dtype
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("attention_forward supports float16 and float32")

    b, h, sq, d = q.shape
    sk = k.shape[2]
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if k.shape[:2] != (b, h) or k.shape[3] != d:
        raise ValueError("q, k, v batch/head/d_head must match")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    scale = d ** -0.5

    # Block sizes — must be >= 16 for tl.dot (tensor-core requirement).
    # Capped at 64 to stay within GPU shared memory limits.
    BLOCK_Q = max(16, min(64, triton.next_power_of_2(sq)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(sk)))
    BLOCK_D = max(16, triton.next_power_of_2(d))

    num_q_blocks = triton.cdiv(sq, BLOCK_Q)
    grid = (b, h, num_q_blocks)

    num_warps = 4 if BLOCK_Q * BLOCK_K <= 4096 else 8

    _attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        sq,
        sk,
        scale,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
