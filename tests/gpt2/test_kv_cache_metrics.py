"""KV cache storage and microbench helpers."""

import pytest
import torch
from transformers import GPT2Config
from transformers.cache_utils import DynamicCache

from benchmarks.gpt2.kv_cache_metrics import (
    kv_cache_storage_nbytes,
    microbench_hf_kv_update_ms_per_decode_step,
    microbench_quant_kv_append_ms_per_decode_step,
    tensor_storage_nbytes,
)
from models.gpt2.kv_cache import QuantizedKVCache, _LayerQuantizedKV


def test_kv_cache_storage_nbytes_none():
    assert kv_cache_storage_nbytes(None) == 0


def test_tensor_storage_nbytes_matches_numel():
    t = torch.zeros(4, 5, dtype=torch.float32)
    assert tensor_storage_nbytes(t) == t.numel() * 4


def test_kv_cache_storage_quantized_matches_manual_sum():
    cache = QuantizedKVCache(group_size=32)
    k_p = torch.zeros(1, 2, 3, 8, dtype=torch.uint8)
    k_s = torch.zeros(1, 2, 3, 2, dtype=torch.float16)
    v_p = torch.ones(1, 2, 3, 8, dtype=torch.uint8)
    v_s = torch.ones(1, 2, 3, 2, dtype=torch.float16)
    cache._layers[0] = _LayerQuantizedKV(
        k_packed=k_p,
        k_scales=k_s,
        v_packed=v_p,
        v_scales=v_s,
        seq_len=3,
    )
    want = (
        tensor_storage_nbytes(k_p)
        + tensor_storage_nbytes(k_s)
        + tensor_storage_nbytes(v_p)
        + tensor_storage_nbytes(v_s)
    )
    assert kv_cache_storage_nbytes(cache) == want


def test_kv_cache_storage_dynamic_cache():
    cache = DynamicCache()
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    cache.update(k, v, 0, {"cache_position": torch.tensor([0])})
    got = kv_cache_storage_nbytes(cache)
    layer = cache.layers[0]
    want = tensor_storage_nbytes(layer.keys) + tensor_storage_nbytes(layer.values)
    assert got == want


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_microbench_hf_and_quant_smoke():
    cfg = GPT2Config(n_layer=2, n_head=4, n_embd=64)
    d = torch.device("cuda")
    a = microbench_hf_kv_update_ms_per_decode_step(
        cfg, d, warmup=1, iters=2, prefill_seq_len=0
    )
    b = microbench_quant_kv_append_ms_per_decode_step(
        cfg, d, group_size=32, warmup=1, iters=2, prefill_seq_len=0
    )
    assert a > 0 and b > 0
    assert a == a and b == b
