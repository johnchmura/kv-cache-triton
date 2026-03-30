"""KV cache tensor storage and cache-update microbenchmarks (not full-model forward)."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers import PretrainedConfig

from models.kv_cache import QuantizedKVCache


def tensor_storage_nbytes(t: torch.Tensor) -> int:
    return int(t.untyped_storage().nbytes())


def _sum_tensors_bytes(tensors: list[torch.Tensor | None]) -> int:
    total = 0
    for t in tensors:
        if t is not None and isinstance(t, torch.Tensor):
            total += tensor_storage_nbytes(t)
    return total


def kv_cache_storage_nbytes(past: Any) -> int:
    """Sum backing storage of tensors holding K/V cache only (not weights or activations)."""
    if past is None:
        return 0
    if isinstance(past, QuantizedKVCache):
        n = 0
        for layer in past._layers.values():
            n += tensor_storage_nbytes(layer.k_packed)
            n += tensor_storage_nbytes(layer.k_scales)
            n += tensor_storage_nbytes(layer.v_packed)
            n += tensor_storage_nbytes(layer.v_scales)
        return n
    layers = getattr(past, "layers", None)
    if layers is not None:
        tensors: list[torch.Tensor | None] = []
        for layer in layers:
            if layer is None:
                continue
            tensors.append(getattr(layer, "keys", None))
            tensors.append(getattr(layer, "values", None))
        return _sum_tensors_bytes(tensors)
    key_cache = getattr(past, "key_cache", None)
    value_cache = getattr(past, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        tensors = [t for t in list(key_cache) + list(value_cache) if t is not None]
        return sum(tensor_storage_nbytes(t) for t in tensors)
    if isinstance(past, (tuple, list)):
        n = 0
        for layer_entry in past:
            if layer_entry is None:
                continue
            if isinstance(layer_entry, (tuple, list)) and len(layer_entry) >= 2:
                k, v = layer_entry[0], layer_entry[1]
                n += _sum_tensors_bytes([k, v])
        return n
    return 0


def _gpt2_kv_shapes(config: PretrainedConfig) -> tuple[int, int, int]:
    n_layer = int(config.n_layer)
    n_head = int(config.n_head)
    head_dim = int(config.n_embd) // n_head
    return n_layer, n_head, head_dim


def _hf_prefill_cache(
    cache: DynamicCache,
    k: torch.Tensor,
    v: torch.Tensor,
    n_layer: int,
    prefill_seq_len: int,
) -> None:
    for layer_idx in range(n_layer):
        for pos in range(prefill_seq_len):
            cache.update(
                k,
                v,
                layer_idx,
                {"cache_position": torch.tensor([pos], device=k.device, dtype=torch.long)},
            )


def _hf_one_decode_cache_update(
    cache: DynamicCache,
    k: torch.Tensor,
    v: torch.Tensor,
    n_layer: int,
    prefill_seq_len: int,
) -> None:
    pos_next = prefill_seq_len
    for layer_idx in range(n_layer):
        cache.update(
            k,
            v,
            layer_idx,
            {"cache_position": torch.tensor([pos_next], device=k.device, dtype=torch.long)},
        )


def microbench_hf_kv_update_ms_per_decode_step(
    config: PretrainedConfig,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
    prefill_seq_len: int,
) -> float:
    """Time DynamicCache.update for all layers once (one decode step); excludes QKV and attention."""
    n_layer, n_head, head_dim = _gpt2_kv_shapes(config)
    k = torch.randn(1, n_head, 1, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(1, n_head, 1, head_dim, device=device, dtype=torch.float16)

    for _ in range(max(1, warmup)):
        c = DynamicCache()
        _hf_prefill_cache(c, k, v, n_layer, prefill_seq_len)
        _hf_one_decode_cache_update(c, k, v, n_layer, prefill_seq_len)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        cache = DynamicCache()
        _hf_prefill_cache(cache, k, v, n_layer, prefill_seq_len)
        start.record()
        _hf_one_decode_cache_update(cache, k, v, n_layer, prefill_seq_len)
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)
    return elapsed_ms / max(1, iters)


def microbench_quant_kv_append_ms_per_decode_step(
    config: PretrainedConfig,
    device: torch.device,
    *,
    group_size: int,
    warmup: int,
    iters: int,
    prefill_seq_len: int,
) -> float:
    """Time QuantizedKVCache.append for all layers once per iter; includes quantize + cat."""
    n_layer, n_head, head_dim = _gpt2_kv_shapes(config)
    k = torch.randn(1, n_head, 1, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(1, n_head, 1, head_dim, device=device, dtype=torch.float16)

    def prefill(cache: QuantizedKVCache) -> None:
        for _ in range(prefill_seq_len):
            for layer_idx in range(n_layer):
                cache.append(k, v, layer_idx)

    def one_append_round(cache: QuantizedKVCache) -> None:
        for layer_idx in range(n_layer):
            cache.append(k, v, layer_idx)

    for _ in range(max(1, warmup)):
        c = QuantizedKVCache(group_size=group_size)
        prefill(c)
        one_append_round(c)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        cache = QuantizedKVCache(group_size=group_size)
        prefill(cache)
        start.record()
        one_append_round(cache)
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)
    return elapsed_ms / max(1, iters)
