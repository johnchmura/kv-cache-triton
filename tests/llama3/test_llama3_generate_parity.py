"""Greedy generate parity on a tiny synthetic Llama config.

Tolerates divergence after a handful of steps (quantization noise compounds)
but requires top-1 agreement for the first few tokens.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.llama3_quant import replace_llama_attention_with_quantized

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tiny_llama(device: torch.device) -> LlamaForCausalLM:
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
    return LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()


def _greedy(model: LlamaForCausalLM, prompt: torch.Tensor, cache, n_new: int) -> list[int]:
    tokens: list[int] = []
    with torch.no_grad():
        out = model(input_ids=prompt, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens.append(int(next_id.item()))
        for _ in range(n_new - 1):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(int(next_id.item()))
    return tokens


@cuda
def test_greedy_agrees_first_steps() -> None:
    device = torch.device("cuda")
    model_ref = _tiny_llama(device)
    model_q = _tiny_llama(device)
    model_q.load_state_dict(model_ref.state_dict())
    replace_llama_attention_with_quantized(model_q, group_size=32)

    torch.manual_seed(321)
    prompt = torch.randint(0, 256, (1, 24), device=device, dtype=torch.long)

    tokens_ref = _greedy(model_ref, prompt, DynamicCache(), n_new=16)
    tokens_q = _greedy(model_q, prompt, QuantizedKVCache(group_size=32), n_new=16)

    agree_prefix = 0
    for a, b in zip(tokens_ref, tokens_q):
        if a == b:
            agree_prefix += 1
        else:
            break
    assert agree_prefix >= 4, (
        f"greedy prefix agreement only {agree_prefix} tokens (ref={tokens_ref}, quant={tokens_q})"
    )
