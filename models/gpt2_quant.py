"""GPT-2 attention with quantized KV cache and fused Triton decode kernel."""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, eager_attention_forward

from kernels.attention_quant import attention_forward_quant
from models.kv_cache import QuantizedKVCache


class GPT2AttentionQuantized(GPT2Attention):
    """Same as GPT2Attention, with quantized KV cache support for decode."""

    def __init__(self, config, is_cross_attention: bool = False, layer_idx: int | None = None, group_size: int = 32):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.quant_group_size = int(group_size)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        quant_cache = past_key_values if isinstance(past_key_values, QuantizedKVCache) else None
        if past_key_values is not None and quant_cache is None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            if past_key_values is not None and quant_cache is None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if quant_cache is not None and not is_cross_attention:
            quant_cache.append(key_states, value_states, self.layer_idx)
            k_packed, k_scales, v_packed, v_scales = quant_cache.get_quantized(self.layer_idx)
        elif (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        use_quant_triton = (
            quant_cache is not None
            and not is_cross_attention
            and query_states.is_cuda
            and not (using_eager and self.reorder_and_upcast_attn)
            and (query_states.shape[-2] == 1 or attention_mask is None)
        )

        if using_eager and self.reorder_and_upcast_attn:
            if quant_cache is not None and not is_cross_attention:
                key_states, value_states = quant_cache.get_dequantized(self.layer_idx)
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        elif use_quant_triton:
            attn_output = attention_forward_quant(
                query_states.contiguous(),
                k_packed.contiguous(),
                k_scales.contiguous(),
                v_packed.contiguous(),
                v_scales.contiguous(),
                group_size=self.quant_group_size,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_weights = None
        else:
            if quant_cache is not None and not is_cross_attention:
                key_states, value_states = quant_cache.get_dequantized(self.layer_idx)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


def replace_gpt2_attention_with_quantized(model: torch.nn.Module, group_size: int = 32) -> None:
    """In-place: swap each GPT2Block self-attention for quantized-cache attention."""
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise ValueError("expected a GPT2 model with .transformer.h blocks")
    for block in blocks:
        old = block.attn
        if isinstance(old, GPT2AttentionQuantized):
            old.quant_group_size = int(group_size)
            continue
        device = next(old.parameters()).device
        ref_dtype = next(old.parameters()).dtype
        new = GPT2AttentionQuantized(old.config, layer_idx=old.layer_idx, group_size=group_size)
        new.load_state_dict(old.state_dict())
        new.to(device=device, dtype=ref_dtype)
        new.train(old.training)
        block.attn = new
