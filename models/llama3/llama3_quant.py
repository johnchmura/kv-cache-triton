"""Llama 3 attention with quantized KV cache and fused Triton decode kernel.

Integration shape:
    hidden_states -> q_proj / k_proj / v_proj
    reshape to [B, n_q_or_kv, S, d_head]
    RoPE on q, k
    if past_key_values is a QuantizedKVCache:
        quant_cache.append(k, v, layer_idx)
        (k_packed, k_scales, v_packed, v_scales) = quant_cache.get_quantized(layer_idx)
        fused attention over packed cache (is_causal selected from q_len / mask)
    else:
        fall back to the stock LlamaAttention path via super().forward
    o_proj

We keep a full fallback path so the same module can be used with a plain
``DynamicCache`` for reference comparisons inside a single process.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from kernels.llama3.attention_quant import attention_forward_quant_gqa
from models.llama3.kv_cache import QuantizedKVCache


class LlamaAttentionQuantized(LlamaAttention):
    """LlamaAttention subclass that routes to a fused INT4 kernel when a ``QuantizedKVCache`` is in use."""

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

        quant_cache = past_key_values if isinstance(past_key_values, QuantizedKVCache) else None

        if quant_cache is not None:
            quant_cache.append(key_states, value_states, self.layer_idx)
            k_packed, k_scales, v_packed, v_scales = quant_cache.get_quantized(self.layer_idx)

            n_q = self.config.num_attention_heads
            n_kv = self.config.num_key_value_heads
            is_causal = attention_mask is None and query_states.shape[-2] > 1
            if query_states.shape[-2] == 1:
                is_causal = False

            if query_states.is_cuda and attention_mask is None:
                attn_output = attention_forward_quant_gqa(
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

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _transplant_projection(src: torch.nn.Linear, dst: torch.nn.Linear) -> None:
    """Move weight/bias tensors from ``src`` to ``dst`` and re-attach any ``accelerate`` hook.

    Works with plain CUDA/CPU parameters and with ``accelerate``-offloaded parameters
    (meta tensors paired with CPU storage behind a hook). We rebind the ``Parameter``
    objects, then re-call ``add_hook_to_module`` on the destination so the forward
    wrapper (which is needed for weight streaming) is installed on the new module.
    """
    dst.weight = src.weight
    if getattr(src, "bias", None) is not None:
        dst.bias = src.bias
    else:
        dst.bias = None

    hook = getattr(src, "_hf_hook", None)
    if hook is not None:
        try:
            from accelerate.hooks import add_hook_to_module, remove_hook_from_module

            remove_hook_from_module(src)
            add_hook_to_module(dst, hook)
        except Exception:
            pass


def replace_llama_attention_with_quantized(
    model: torch.nn.Module, group_size: int = 32
) -> None:
    """In-place: swap each Llama decoder-layer self-attention for the quantized variant.

    Preserves any ``accelerate`` offload hooks on the inner q/k/v/o projections, so
    the same model loaded with ``device_map='auto'`` continues to work (parameters
    that live on CPU keep their hook-based streaming).
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
        if isinstance(old, LlamaAttentionQuantized):
            old.quant_group_size = int(group_size)
            continue
        new = LlamaAttentionQuantized(old.config, layer_idx=old.layer_idx, group_size=group_size)
        for pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _transplant_projection(getattr(old, pname), getattr(new, pname))

        old_hook = getattr(old, "_hf_hook", None)
        if _ACCEL and old_hook is not None:
            remove_hook_from_module(old)
            add_hook_to_module(new, old_hook)
        new.train(old.training)
        layer.self_attn = new
