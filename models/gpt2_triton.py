"""GPT-2 attention with Triton single-query attention on CUDA decode steps (q_len == 1)."""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, eager_attention_forward

from kernels.attention import attention_forward


class GPT2AttentionTriton(GPT2Attention):
    """Same as GPT2Attention; uses Triton for attention when query length is 1 (decode)."""

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
        if past_key_values is not None:
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

            if past_key_values is not None and is_updated:
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

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            from kernels.quantization import quantize_and_pack_int4, unpack_and_dequantize_int4
            if not hasattr(curr_past_key_value, "int4_k_scales"):
                curr_past_key_value.int4_k_scales = {}
                curr_past_key_value.int4_v_scales = {}

            pk_new, sk_new = quantize_and_pack_int4(key_states, dim=-1)
            pv_new, sv_new = quantize_and_pack_int4(value_states, dim=-1)

            is_prefill = curr_past_key_value.get_seq_length(self.layer_idx) == 0

            pk, pv = curr_past_key_value.update(pk_new, pv_new, self.layer_idx, cache_kwargs=kwargs.get("cache_position"))

            if is_prefill:
                curr_past_key_value.int4_k_scales[self.layer_idx] = sk_new
                curr_past_key_value.int4_v_scales[self.layer_idx] = sv_new
            else:
                sk = torch.cat([curr_past_key_value.int4_k_scales[self.layer_idx], sk_new], dim=-2)
                sv = torch.cat([curr_past_key_value.int4_v_scales[self.layer_idx], sv_new], dim=-2)
                curr_past_key_value.int4_k_scales[self.layer_idx] = sk
                curr_past_key_value.int4_v_scales[self.layer_idx] = sv

            sk_full = curr_past_key_value.int4_k_scales[self.layer_idx]
            sv_full = curr_past_key_value.int4_v_scales[self.layer_idx]
            
            key_states = unpack_and_dequantize_int4(pk, sk_full, dim=-1)
            value_states = unpack_and_dequantize_int4(pv, sv_full, dim=-1)

            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        use_triton = (
            not is_cross_attention
            and query_states.is_cuda
            and not (using_eager and self.reorder_and_upcast_attn)
            and (query_states.shape[-2] > 1)
            and (attention_mask is None)
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        elif use_triton:
            if query_states.shape[-2] == 1 and past_key_values is not None:
                attn_output = attention_forward(
                    query_states.contiguous(),
                    pk.contiguous(),
                    pv.contiguous(),
                    is_causal=False,
                    k_scales=sk_full,
                    v_scales=sv_full,
                )
            else:
                attn_output = attention_forward(
                    query_states.contiguous(),
                    key_states.contiguous(),
                    value_states.contiguous(),
                    is_causal=is_causal,
                )
            attn_output = attn_output.transpose(1, 2)
            attn_weights = None
        else:
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


def replace_gpt2_attention_with_triton(model: torch.nn.Module) -> None:
    """In-place: swap each GPT2Block self-attention with GPT2AttentionTriton (weights preserved)."""
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise ValueError("expected a GPT2 model with .transformer.h blocks")
    for block in blocks:
        old = block.attn
        if isinstance(old, GPT2AttentionTriton):
            continue
        device = next(old.parameters()).device
        ref_dtype = next(old.parameters()).dtype
        new = GPT2AttentionTriton(old.config, layer_idx=old.layer_idx)
        new.load_state_dict(old.state_dict())
        new.to(device=device, dtype=ref_dtype)
        new.train(old.training)
        block.attn = new
