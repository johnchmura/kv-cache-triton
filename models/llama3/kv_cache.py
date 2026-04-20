"""Quantized KV cache for Llama 3.1 decode with packed INT4 storage.

Stores ``[B, n_kv, S, d_pack]`` packed K/V plus ``[B, n_kv, S, n_groups]`` scales
per layer. Duck-types the minimal subset of the transformers ``Cache`` API that
the Llama attention path actually calls: ``get_seq_length``, ``get_mask_sizes``,
``get_max_cache_shape``, ``update`` and ``reset``. The attention module peeks at
the ``QuantizedKVCache`` directly to grab packed pointers for the fused kernel,
so ``update`` only has to return something tensor-shaped for fallback code
paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.llama3.quantize import dequantize_int4, quantize_int4


@dataclass
class _LayerQuantizedKV:
    k_packed: torch.Tensor
    k_scales: torch.Tensor
    v_packed: torch.Tensor
    v_scales: torch.Tensor
    seq_len: int


class QuantizedKVCache:
    """Per-layer INT4-packed KV cache. Compatible with HF Llama attention."""

    def __init__(self, group_size: int = 32):
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        self.group_size = int(group_size)
        self._layers: dict[int, _LayerQuantizedKV] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        layer = self._layers.get(layer_idx)
        return 0 if layer is None else int(layer.seq_len)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        past_len = self.get_seq_length(layer_idx)
        query_length = int(cache_position.shape[0])
        return past_len + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    @property
    def is_initialized(self) -> bool:
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        return [False for _ in self._layers]

    def __len__(self) -> int:
        return len(self._layers)

    def reset(self) -> None:
        self._layers.clear()

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._layers

    def _append_quantized(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
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

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        self._append_quantized(key_states, value_states, layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and append K/V, then return the full dequantized cache.

        The fused attention path reads packed tensors directly via
        ``get_quantized`` and never uses this return value. It exists for
        compatibility with the stock Llama attention fallback.
        """
        del cache_kwargs
        self._append_quantized(key_states, value_states, layer_idx)
        return self.get_dequantized(layer_idx, out_dtype=key_states.dtype)

    def get_quantized(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self._layers.get(layer_idx)
        if layer is None:
            raise KeyError(f"no cache state for layer {layer_idx}")
        return layer.k_packed, layer.k_scales, layer.v_packed, layer.v_scales

    def get_dequantized(
        self,
        layer_idx: int,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._layers.get(layer_idx)
        if layer is None:
            raise KeyError(f"no cache state for layer {layer_idx}")

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

    def nbytes(self) -> int:
        """Total bytes stored across all layers (packed + scales)."""
        total = 0
        for layer in self._layers.values():
            total += layer.k_packed.element_size() * layer.k_packed.numel()
            total += layer.v_packed.element_size() * layer.v_packed.numel()
            total += layer.k_scales.element_size() * layer.k_scales.numel()
            total += layer.v_scales.element_size() * layer.v_scales.numel()
        return total
