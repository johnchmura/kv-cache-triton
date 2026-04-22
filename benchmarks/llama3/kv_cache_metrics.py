"""KV cache sizing helpers for both HF DynamicCache and QuantizedKVCache."""

from __future__ import annotations

import torch

# NOTE: kv_cache implementations are kernel-variant dependent (default vs sm70).
# Avoid hard `isinstance` checks against a single class; prefer capability checks.
from models.llama3.kv_cache import QuantizedKVCache


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def kv_cache_nbytes(past) -> int:
    """Total bytes used by ``past`` for its cached K/V tensors.

    Handles both ``QuantizedKVCache`` (our own path) and the generic transformers
    ``Cache`` interface (DynamicCache and its friends). Newer ``DynamicCache``
    iterations may yield tuples longer than ``(key, value)``; we prefer
    ``key_cache`` / ``value_cache`` when present.
    """
    if past is None:
        return 0
    # Support both default and sm70-bound QuantizedKVCache classes.
    if hasattr(past, "nbytes") and callable(getattr(past, "nbytes", None)):
        try:
            return int(past.nbytes())
        except Exception:
            pass
    if isinstance(past, QuantizedKVCache):
        try:
            return int(past.nbytes())
        except Exception:
            return 0

    key_cache = getattr(past, "key_cache", None)
    value_cache = getattr(past, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        total = 0
        for t in list(key_cache) + list(value_cache):
            if t is not None and isinstance(t, torch.Tensor):
                total += _tensor_nbytes(t)
        return total

    layers = getattr(past, "layers", None)
    if layers is not None:
        total = 0
        for layer in layers:
            if layer is None:
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                total += _tensor_nbytes(keys)
            if isinstance(values, torch.Tensor):
                total += _tensor_nbytes(values)
        return total

    total = 0
    try:
        for layer_entry in past:
            if layer_entry is None:
                continue
            if isinstance(layer_entry, (tuple, list)) and len(layer_entry) >= 2:
                k, v = layer_entry[0], layer_entry[1]
                if isinstance(k, torch.Tensor):
                    total += _tensor_nbytes(k)
                if isinstance(v, torch.Tensor):
                    total += _tensor_nbytes(v)
    except Exception:
        return 0
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
