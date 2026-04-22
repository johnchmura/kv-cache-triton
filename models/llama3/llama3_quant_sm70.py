"""Volta-safe (sm_70) Llama 3 attention with quantized KV cache.

Mirrors :mod:`models.llama3.llama3_quant` but uses the sm_70-safe fused
attention kernel and the sm_70-safe quantized KV cache. The only behavioral
difference is at the Triton boundary: bf16 tensors never reach the kernel.

Diagnostic env vars:

- ``KV_FORCE_DEQUANT_FALLBACK=1`` — bypass the fused kernel; use ``get_dequantized`` + HF attention.
- ``KV_DEBUG_ATTENTION_ROUTE=1`` — print per-layer mask info once; print layer0 fused vs dequant SDPA once.
- ``KV_DEBUG_ROPED_KV_LAYER=<idx>`` or ``KV_DEBUG_LAYER0_ROPED_KV=1`` — capture post-RoPE K/V for roundtrip scripts.
- ``KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1`` — capture post-RoPE Q for every layer (used by fused-vs-dequant diff).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from kernels.llama3.attention_quant_sm70 import attention_forward_quant_gqa_sm70
from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.kv_cache_sm70 import QuantizedKVCacheSM70
from models.llama3.llama3_quant import _transplant_projection

_FORCE_DEQUANT_FALLBACK = os.environ.get("KV_FORCE_DEQUANT_FALLBACK", "") == "1"

# ``KV_DEBUG_ATTENTION_ROUTE=1``: once per layer, log mask / fused vs fallback; optional device check.

_ATTN_ROUTE_LOGGED_LAYERS: set[int] = set()
_LAYER0_ROUTE_BRANCH_PRINTED: bool = False

# RoPE K/V capture for roundtrip scripts: ``KV_DEBUG_LAYER0_ROPED_KV=1`` (layer 0) or
# ``KV_DEBUG_ROPED_KV_LAYER=<idx>``.
_DEBUG_ROPED_KV: tuple[torch.Tensor, torch.Tensor] | None = None
_DEBUG_ROPED_KV_LAYER: int | None = None

# Post-RoPE Q captures keyed by layer_idx. Populated only when
# KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1. Consumers (scripts/llama3_logits_diff.py)
# must drain these via get_debug_captured_q / clear_debug_captured_q; the
# tensors are kept on the model device and can be large.
_DEBUG_CAPTURED_Q: dict[int, torch.Tensor] = {}


def _debug_roped_kv_target_layer() -> int | None:
    if os.environ.get("KV_DEBUG_LAYER0_ROPED_KV", "") == "1":
        return 0
    s = os.environ.get("KV_DEBUG_ROPED_KV_LAYER", "").strip()
    if s.isdigit():
        return int(s)
    return None


def get_debug_layer0_roped_kv() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Backward-compatible name; returns capture for the configured debug layer."""
    return _DEBUG_ROPED_KV


def get_debug_roped_kv_layer_idx() -> int | None:
    return _DEBUG_ROPED_KV_LAYER


def clear_debug_layer0_roped_kv() -> None:
    global _DEBUG_ROPED_KV, _DEBUG_ROPED_KV_LAYER
    _DEBUG_ROPED_KV = None
    _DEBUG_ROPED_KV_LAYER = None


def get_debug_captured_q() -> dict[int, torch.Tensor]:
    """Post-RoPE Q tensors captured under KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1."""
    return dict(_DEBUG_CAPTURED_Q)


def clear_debug_captured_q() -> None:
    _DEBUG_CAPTURED_Q.clear()


class LlamaAttentionQuantizedSM70(LlamaAttention):
    """LlamaAttention variant that routes to the sm_70-safe INT4 kernel."""

    def __init__(self, config, layer_idx: int, group_size: int = 32):
        super().__init__(config=config, layer_idx=layer_idx)
        self.quant_group_size = int(group_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _cap_layer = _debug_roped_kv_target_layer()
        if _cap_layer is not None and self.layer_idx == _cap_layer:
            global _DEBUG_ROPED_KV, _DEBUG_ROPED_KV_LAYER
            _DEBUG_ROPED_KV = (
                key_states.detach().clone(),
                value_states.detach().clone(),
            )
            _DEBUG_ROPED_KV_LAYER = self.layer_idx

        if os.environ.get("KV_DEBUG_CAPTURE_Q_ALL_LAYERS", "") == "1":
            _DEBUG_CAPTURED_Q[int(self.layer_idx)] = query_states.detach().clone()

        quant_cache = past_key_values if isinstance(past_key_values, QuantizedKVCache) else None

        if quant_cache is not None:
            quant_cache.append(key_states, value_states, self.layer_idx)

            n_q = self.config.num_attention_heads
            n_kv = self.config.num_key_value_heads
            is_causal = attention_mask is None and query_states.shape[-2] > 1
            if query_states.shape[-2] == 1:
                is_causal = False

            _dbg_route = os.environ.get("KV_DEBUG_ATTENTION_ROUTE", "") == "1"
            if _dbg_route and self.layer_idx not in _ATTN_ROUTE_LOGGED_LAYERS:
                _ATTN_ROUTE_LOGGED_LAYERS.add(self.layer_idx)
                mask_desc = "None" if attention_mask is None else f"tensor shape={tuple(attention_mask.shape)} dtype={attention_mask.dtype}"
                print(
                    f"[KV_DEBUG_ATTENTION_ROUTE] layer={self.layer_idx} attention_mask={mask_desc} "
                    f"sq={query_states.shape[-2]} force_dequant_fallback={_FORCE_DEQUANT_FALLBACK}",
                    flush=True,
                )

            _fused_ok = (
                query_states.is_cuda
                and attention_mask is None
                and not _FORCE_DEQUANT_FALLBACK
            )
            global _LAYER0_ROUTE_BRANCH_PRINTED
            if _dbg_route and self.layer_idx == 0 and not _LAYER0_ROUTE_BRANCH_PRINTED:
                _LAYER0_ROUTE_BRANCH_PRINTED = True
                print(
                    f"[KV_DEBUG_ATTENTION_ROUTE] layer0 branch={'fused_sm70' if _fused_ok else 'dequant_SDPA'}",
                    flush=True,
                )

            if _fused_ok:
                k_packed, k_scales, v_packed, v_scales = quant_cache.get_quantized(self.layer_idx)
                if _dbg_route:
                    qd = query_states.device
                    if k_packed.device != qd or v_packed.device != qd:
                        print(
                            f"[KV_DEBUG_ATTENTION_ROUTE] device mismatch: q={qd} k_packed={k_packed.device} v_packed={v_packed.device}",
                            flush=True,
                        )
                attn_output = attention_forward_quant_gqa_sm70(
                    query_states.contiguous(),
                    k_packed.contiguous(),
                    k_scales.contiguous(),
                    v_packed.contiguous(),
                    v_scales.contiguous(),
                    n_q_heads=n_q,
                    n_kv_heads=n_kv,
                    group_size=self.quant_group_size,
                    is_causal=is_causal,
                )
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None

            k_full, v_full = quant_cache.get_dequantized(
                self.layer_idx, out_dtype=query_states.dtype
            )
            if _dbg_route and k_full.device != query_states.device:
                print(
                    f"[KV_DEBUG_ATTENTION_ROUTE] layer={self.layer_idx} dequant K/V device={k_full.device} "
                    f"q={query_states.device}",
                    flush=True,
                )
            return self._fallback_attention(
                query_states,
                k_full,
                v_full,
                attention_mask=attention_mask,
                input_shape=input_shape,
                **kwargs,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        return self._fallback_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            input_shape=input_shape,
            **kwargs,
        )

    def _fallback_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, ...],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        out_dtype = query_states.dtype
        q = query_states
        k = key_states
        v = value_states
        # Eager path does Q@K^T in the activations dtype; bf16 range is tight for long
        # sequences and INT4 dequant noise. fp32 scores match HF softmax (fp32) staging.
        if self.config._attn_implementation == "eager" and out_dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        if attn_output.dtype != out_dtype:
            attn_output = attn_output.to(out_dtype)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def replace_llama_attention_with_quantized_sm70(
    model: torch.nn.Module, group_size: int = 32
) -> None:
    """In-place: swap each Llama decoder-layer self-attention for the sm_70 variant.

    Preserves any ``accelerate`` offload hooks on the inner q/k/v/o projections
    the same way :func:`models.llama3.llama3_quant.replace_llama_attention_with_quantized`
    does.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("expected a Llama CausalLM model with .model.layers")

    try:
        from accelerate.hooks import add_hook_to_module, remove_hook_from_module
        _ACCEL = True
    except Exception:
        _ACCEL = False

    for layer in layers:
        old = layer.self_attn
        if isinstance(old, LlamaAttentionQuantizedSM70):
            old.quant_group_size = int(group_size)
            continue
        new = LlamaAttentionQuantizedSM70(
            old.config, layer_idx=old.layer_idx, group_size=group_size
        )
        for pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _transplant_projection(getattr(old, pname), getattr(new, pname))

        old_hook = getattr(old, "_hf_hook", None)
        if _ACCEL and old_hook is not None:
            remove_hook_from_module(old)
            add_hook_to_module(new, old_hook)
        new.train(old.training)
        layer.self_attn = new


__all__ = [
    "LlamaAttentionQuantizedSM70",
    "QuantizedKVCacheSM70",
    "replace_llama_attention_with_quantized_sm70",
    "get_debug_layer0_roped_kv",
    "get_debug_roped_kv_layer_idx",
    "clear_debug_layer0_roped_kv",
    "get_debug_captured_q",
    "clear_debug_captured_q",
]
