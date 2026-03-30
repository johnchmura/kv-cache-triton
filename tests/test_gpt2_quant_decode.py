"""GPT-2 prefill/decode parity with quantized KV cache."""

import pytest
import torch
from transformers import GPT2LMHeadModel

from models.gpt2_quant import replace_gpt2_attention_with_quantized
from models.kv_cache import QuantizedKVCache

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_gpt2_decode_steps_quantized_cache_matches_reference_fp16():
    dtype = torch.float16
    ref = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    quant = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    replace_gpt2_attention_with_quantized(quant, group_size=32)

    torch.manual_seed(1)
    input_ids = torch.randint(0, 50257, (1, 24), device="cuda", dtype=torch.long)

    with torch.no_grad():
        out_ref = ref(input_ids=input_ids, use_cache=True)
        q_cache = QuantizedKVCache(group_size=32)
        out_quant = quant(input_ids=input_ids, use_cache=True, past_key_values=q_cache)

    assert torch.isfinite(out_quant.logits).all()
    prefill_mean_abs = (out_ref.logits - out_quant.logits).abs().mean().item()
    assert prefill_mean_abs < 10.0

    past_ref = out_ref.past_key_values
    for _ in range(6):
        next_tok = torch.randint(0, 50257, (1, 1), device="cuda", dtype=torch.long)
        with torch.no_grad():
            step_ref = ref(input_ids=next_tok, past_key_values=past_ref, use_cache=True)
            step_quant = quant(input_ids=next_tok, past_key_values=q_cache, use_cache=True)

        assert torch.isfinite(step_quant.logits).all()
        step_mean_abs = (step_ref.logits - step_quant.logits).abs().mean().item()
        assert step_mean_abs < 12.0
        past_ref = step_ref.past_key_values
