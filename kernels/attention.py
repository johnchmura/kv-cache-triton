"""Single-query scaled dot-product attention: Q [B,H,1,D], K/V [B,H,S,D]."""

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
    stride_q_1,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_1,
    stride_o_d,
    seq_len,
    d_head,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    o_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_head
    q = tl.load(q_base + offs_d * stride_q_d, mask=mask_d, other=0.0)
    q = q.to(tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < seq_len
    s_idx = offs_s[:, None]
    d_idx = offs_d[None, :]
    k_ptrs = k_base + s_idx * stride_k_s + d_idx * stride_k_d
    mask_k = mask_s[:, None] & mask_d[None, :]
    k_tile = tl.load(k_ptrs, mask=mask_k, other=0.0)
    k_tile = k_tile.to(tl.float32)

    logits = tl.sum(q[None, :] * k_tile, axis=1) * scale
    logits = tl.where(mask_s, logits, float("-inf"))

    m = tl.max(logits)
    logits = logits - m
    w = tl.exp(logits)
    denom = tl.sum(w)
    attn = w / denom

    v_ptrs = v_base + offs_s[:, None] * stride_v_s + d_idx * stride_v_d
    v_tile = tl.load(v_ptrs, mask=mask_k, other=0.0)
    v_tile = v_tile.to(tl.float32)
    out_vec = tl.sum(attn[:, None] * v_tile, axis=0)

    if IS_FP16:
        tl.store(o_base + offs_d * stride_o_d, out_vec.to(tl.float16), mask=mask_d)
    else:
        tl.store(o_base + offs_d * stride_o_d, out_vec, mask=mask_d)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention for a single query position per head.

    Shapes: q [B, H, 1, D], k and v [B, H, S, D]. Returns [B, H, 1, D].
    """
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("attention_forward expects CUDA tensors")
    dtype = q.dtype
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("attention_forward supports float16 and float32")

    b, h, sq, d = q.shape
    if sq != 1:
        raise ValueError(f"q must have sequence length 1, got {sq}")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if k.shape[:2] != (b, h) or k.shape[3] != d:
        raise ValueError("q, k, v batch/head/d_head must match")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    seq_len = k.shape[2]
    d_head = d
    scale = d_head**-0.5

    block_s = triton.next_power_of_2(seq_len)
    block_d = triton.next_power_of_2(d_head)
    is_fp16 = dtype == torch.float16

    grid = (b, h)
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
        seq_len,
        d_head,
        scale,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        IS_FP16=is_fp16,
    )
    return out
