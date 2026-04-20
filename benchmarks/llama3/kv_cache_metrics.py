"""KV cache sizing helpers for both HF DynamicCache and QuantizedKVCache."""

from __future__ import annotations

from typing import Iterable

import torch

from models.llama3.kv_cache import QuantizedKVCache


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def kv_cache_nbytes(past) -> int:
    """Total bytes used by ``past`` for its cached K/V tensors.

    Handles both ``QuantizedKVCache`` (our own path) and the generic transformers
    ``Cache`` interface (DynamicCache and its friends).
    """
    if past is None:
        return 0
    if isinstance(past, QuantizedKVCache):
        return past.nbytes()

    total = 0
    try:
        for keys, values in past:
            if isinstance(keys, torch.Tensor):
                total += _tensor_nbytes(keys)
            if isinstance(values, torch.Tensor):
                total += _tensor_nbytes(values)
    except TypeError:
        layers = getattr(past, "layers", None)
        if layers is None:
            return 0
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                total += _tensor_nbytes(keys)
            if isinstance(values, torch.Tensor):
                total += _tensor_nbytes(values)
    return total


def sum_layer_seq_len(past, n_layers: int) -> int:
    """Rough sanity: total K-seq tokens summed across layers (or 0 if unknown)."""
    if past is None:
        return 0
    if isinstance(past, QuantizedKVCache):
        return sum(past.get_seq_length(i) for i in range(n_layers))
    try:
        return int(past.get_seq_length(0)) * n_layers
    except Exception:
        return 0


def summarize_cache(past, n_layers: int) -> dict:
    return {
        "kv_bytes": kv_cache_nbytes(past),
        "total_seq_tokens": sum_layer_seq_len(past, n_layers),
    }
