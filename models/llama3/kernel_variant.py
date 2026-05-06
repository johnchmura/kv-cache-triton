"""Resolve and bind Llama INT4 KV implementation (default sm_80+ vs Volta sm70)."""

from __future__ import annotations

from collections.abc import Callable

import torch


def resolve_kernel_variant(choice: str) -> str:
    """``auto`` -> ``sm70`` if any visible GPU has compute capability < 8, else ``default``."""
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        return "default"
    for d in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(d)[0]
        if major < 8:
            return "sm70"
    return "default"


def bind_quantized_kernel_variant(variant: str) -> tuple[type, Callable[..., None]]:
    """Return ``(QuantizedKVCache class, replace_llama_attention_with_quantized)`` for ``variant``."""
    if variant == "sm70":
        from models.llama3.kv_cache_sm70 import QuantizedKVCacheSM70 as cache_cls
        from models.llama3.llama3_quant_sm70 import (
            replace_llama_attention_with_quantized_sm70 as replace_fn,
        )
    else:
        from models.llama3.kv_cache import QuantizedKVCache as cache_cls
        from models.llama3.llama3_quant import (
            replace_llama_attention_with_quantized as replace_fn,
        )
    return cache_cls, replace_fn
