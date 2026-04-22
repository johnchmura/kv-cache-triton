"""Volta-safe (sm_70) quantized KV cache for Llama 3.1 decode.

Subclass of :class:`models.llama3.kv_cache.QuantizedKVCache` that routes all
quantize / dequantize calls through the sm_70-safe kernels in
``kernels/llama3/quantize_sm70.py``. Scales are stored in fp32 so the fused
attention kernel never has to load a bf16 tensor on Volta.

Diagnostic: set ``KV_PASSTHROUGH_BF16=1`` in the environment to turn the cache
into a bf16 passthrough (no quantization). Used to bisect whether correctness
regressions come from the INT4 round-trip or from the attention-module wiring
around it. The fused kernel path still wants INT4 packed tensors, so the
passthrough mode only makes sense in combination with
``KV_FORCE_DEQUANT_FALLBACK=1``.

Downstream code can continue to use ``isinstance(cache, QuantizedKVCache)``
because this class inherits from the base cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from kernels.llama3.quantize_sm70 import dequantize_int4, quantize_int4
from models.llama3.kv_cache import QuantizedKVCache, _LayerQuantizedKV

_PASSTHROUGH_BF16 = os.environ.get("KV_PASSTHROUGH_BF16", "") == "1"


@dataclass
class _LayerPassthroughKV:
    k: torch.Tensor
    v: torch.Tensor
    seq_len: int


class QuantizedKVCacheSM70(QuantizedKVCache):
    """Per-layer INT4-packed KV cache with fp32 scales for sm_70 compatibility."""

    def __init__(self, group_size: int = 32):
        super().__init__(group_size=group_size)
        # Populated only when KV_PASSTHROUGH_BF16=1. Kept separate from the
        # INT4 state so both data layouts can coexist in code without type
        # punning the _LayerQuantizedKV dataclass.
        self._passthrough: dict[int, _LayerPassthroughKV] = {}

    def _append_quantized(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if _PASSTHROUGH_BF16:
            prev = self._passthrough.get(layer_idx)
            if prev is None:
                self._passthrough[layer_idx] = _LayerPassthroughKV(
                    k=key_states,
                    v=value_states,
                    seq_len=int(key_states.shape[2]),
                )
                return
            self._passthrough[layer_idx] = _LayerPassthroughKV(
                k=torch.cat([prev.k, key_states], dim=2),
                v=torch.cat([prev.v, value_states], dim=2),
                seq_len=prev.seq_len + int(key_states.shape[2]),
            )
            return

        k_packed_new, k_scales_new = quantize_int4(key_states, group_size=self.group_size)
        v_packed_new, v_scales_new = quantize_int4(value_states, group_size=self.group_size)

        prev = self._layers.get(layer_idx)
        if prev is None:
            self._layers[layer_idx] = _LayerQuantizedKV(
                k_packed=k_packed_new,
                k_scales=k_scales_new,
                v_packed=v_packed_new,
                v_scales=v_scales_new,
                seq_len=int(key_states.shape[2]),
            )
            return

        self._layers[layer_idx] = _LayerQuantizedKV(
            k_packed=torch.cat([prev.k_packed, k_packed_new], dim=2),
            k_scales=torch.cat([prev.k_scales, k_scales_new], dim=2),
            v_packed=torch.cat([prev.v_packed, v_packed_new], dim=2),
            v_scales=torch.cat([prev.v_scales, v_scales_new], dim=2),
            seq_len=prev.seq_len + int(key_states.shape[2]),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if _PASSTHROUGH_BF16:
            layer = self._passthrough.get(layer_idx)
            return 0 if layer is None else int(layer.seq_len)
        return super().get_seq_length(layer_idx)

    def has_layer(self, layer_idx: int) -> bool:
        if _PASSTHROUGH_BF16:
            return layer_idx in self._passthrough
        return super().has_layer(layer_idx)

    def reset(self) -> None:
        self._passthrough.clear()
        super().reset()

    def __len__(self) -> int:
        if _PASSTHROUGH_BF16:
            return len(self._passthrough)
        return super().__len__()

    @property
    def is_initialized(self) -> bool:
        if _PASSTHROUGH_BF16:
            return len(self._passthrough) > 0
        return super().is_initialized

    @property
    def is_sliding(self) -> list[bool]:
        if _PASSTHROUGH_BF16:
            return [False for _ in self._passthrough]
        return super().is_sliding

    def get_dequantized(
        self,
        layer_idx: int,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _PASSTHROUGH_BF16:
            layer = self._passthrough.get(layer_idx)
            if layer is None:
                raise KeyError(f"no cache state for layer {layer_idx}")
            k = layer.k if layer.k.dtype == out_dtype else layer.k.to(out_dtype)
            v = layer.v if layer.v.dtype == out_dtype else layer.v.to(out_dtype)
            return k, v

        layer = self._layers.get(layer_idx)
        if layer is None:
            raise KeyError(f"no cache state for layer {layer_idx}")

        # Guard against a packed/scales width drift after torch.cat on dim=2.
        # The Triton dequant kernel walks packed width as num_groups * (group_size // 2),
        # so a mismatch here would silently produce garbage (and has been blamed
        # for non-finite K/V in past runs).
        expected_packed = layer.k_scales.shape[-1] * (self.group_size // 2)
        if layer.k_packed.shape[-1] != expected_packed:
            raise RuntimeError(
                f"layer {layer_idx}: k_packed width {layer.k_packed.shape[-1]} != "
                f"k_scales groups {layer.k_scales.shape[-1]} * (group_size/2={self.group_size // 2})"
            )
        if layer.v_packed.shape[-1] != layer.v_scales.shape[-1] * (self.group_size // 2):
            raise RuntimeError(
                f"layer {layer_idx}: v_packed width {layer.v_packed.shape[-1]} != "
                f"v_scales groups {layer.v_scales.shape[-1]} * (group_size/2={self.group_size // 2})"
            )

        d_head = layer.k_scales.shape[-1] * self.group_size
        k = dequantize_int4(
            layer.k_packed,
            layer.k_scales,
            group_size=self.group_size,
            d_head=d_head,
            out_dtype=out_dtype,
        )
        v = dequantize_int4(
            layer.v_packed,
            layer.v_scales,
            group_size=self.group_size,
            d_head=d_head,
            out_dtype=out_dtype,
        )
        return k, v
