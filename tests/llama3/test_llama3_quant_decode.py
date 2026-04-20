"""Integration test: tiny Llama config, baseline vs quantized KV decode.

Builds a small ``LlamaForCausalLM`` from random weights, runs the same
prefill + decode sequence through both the stock ``DynamicCache`` path and the
``QuantizedKVCache`` path, and checks last-logit agreement.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.llama3_quant import replace_llama_attention_with_quantized

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tiny_llama(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return model


@cuda
def test_prefill_plus_decode_matches_reference() -> None:
    device = torch.device("cuda")
    model_ref = _tiny_llama(device)
    model_q = _tiny_llama(device)
    model_q.load_state_dict(model_ref.state_dict())
    replace_llama_attention_with_quantized(model_q, group_size=32)

    torch.manual_seed(123)
    prompt = torch.randint(0, 256, (1, 32), device=device, dtype=torch.long)

    with torch.no_grad():
        cache_ref = DynamicCache()
        out_ref = model_ref(input_ids=prompt, past_key_values=cache_ref, use_cache=True)

        cache_q = QuantizedKVCache(group_size=32)
        out_q = model_q(input_ids=prompt, past_key_values=cache_q, use_cache=True)

        logits_ref = out_ref.logits[:, -1, :]
        logits_q = out_q.logits[:, -1, :]
        diff_prefill = (logits_ref.float() - logits_q.float()).abs().max().item()
        assert diff_prefill < 1.0, f"prefill last-logit diff too large: {diff_prefill}"
        assert logits_ref.argmax(-1).item() == logits_q.argmax(-1).item()

        for step in range(8):
            next_id = torch.randint(0, 256, (1, 1), device=device, dtype=torch.long)
            out_ref = model_ref(
                input_ids=next_id,
                past_key_values=cache_ref,
                use_cache=True,
            )
            out_q = model_q(
                input_ids=next_id,
                past_key_values=cache_q,
                use_cache=True,
            )
            logits_ref = out_ref.logits[:, -1, :]
            logits_q = out_q.logits[:, -1, :]
            diff = (logits_ref.float() - logits_q.float()).abs().max().item()
            assert diff < 2.0, f"decode step {step} last-logit diff too large: {diff}"


@cuda
def test_quantized_cache_tracks_seq_len() -> None:
    device = torch.device("cuda")
    model = _tiny_llama(device)
    replace_llama_attention_with_quantized(model, group_size=32)

    prompt = torch.randint(0, 256, (1, 16), device=device, dtype=torch.long)
    with torch.no_grad():
        cache = QuantizedKVCache(group_size=32)
        model(input_ids=prompt, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length(0) == 16
        for step in range(4):
            tok = torch.randint(0, 256, (1, 1), device=device, dtype=torch.long)
            model(input_ids=tok, past_key_values=cache, use_cache=True)
            assert cache.get_seq_length(0) == 16 + step + 1

    assert cache.nbytes() > 0


@cuda
def test_quant_cache_kv_bytes_smaller() -> None:
    device = torch.device("cuda")
    model = _tiny_llama(device, dtype=torch.bfloat16)
    replace_llama_attention_with_quantized(model, group_size=32)

    seq = 128
    prompt = torch.randint(0, 256, (1, seq), device=device, dtype=torch.long)
    with torch.no_grad():
        cache = QuantizedKVCache(group_size=32)
        model(input_ids=prompt, past_key_values=cache, use_cache=True)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    d_head = cfg.head_dim
    baseline_bytes = 2 * num_layers * 1 * n_kv * seq * d_head * 2
    q_bytes = cache.nbytes()
    assert q_bytes < baseline_bytes / 2, (
        f"quantized cache {q_bytes} not < half of baseline {baseline_bytes}"
    )
