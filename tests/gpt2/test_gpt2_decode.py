"""GPT-2 prefill + decode: Triton attention matches reference (FP16)."""

import pytest
import torch
from transformers import GPT2LMHeadModel

from models.gpt2.gpt2_triton import replace_gpt2_attention_with_triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_gpt2_prefill_matches_reference_fp16():
    dtype = torch.float16
    ref = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    tr = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    replace_gpt2_attention_with_triton(tr)

    torch.manual_seed(0)
    input_ids = torch.randint(0, 50257, (1, 32), device="cuda", dtype=torch.long)

    with torch.no_grad():
        out_ref = ref(input_ids=input_ids, use_cache=True)
        out_tr = tr(input_ids=input_ids, use_cache=True)

    assert torch.isfinite(out_tr.logits).all()
    assert torch.allclose(out_ref.logits, out_tr.logits, rtol=1e-2, atol=1e-2)


@cuda
def test_gpt2_decode_steps_match_reference_fp16():
    dtype = torch.float16
    ref = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    tr = GPT2LMHeadModel.from_pretrained("gpt2").cuda().to(dtype).eval()
    replace_gpt2_attention_with_triton(tr)

    torch.manual_seed(1)
    input_ids = torch.randint(0, 50257, (1, 24), device="cuda", dtype=torch.long)

    with torch.no_grad():
        out_ref = ref(input_ids=input_ids, use_cache=True)
        out_tr = tr(input_ids=input_ids, use_cache=True)

    assert torch.allclose(out_ref.logits, out_tr.logits, rtol=1e-2, atol=1e-2)

    past_ref = out_ref.past_key_values
    past_tr = out_tr.past_key_values

    for step in range(8):
        next_tok = torch.randint(0, 50257, (1, 1), device="cuda", dtype=torch.long)
        with torch.no_grad():
            step_ref = ref(input_ids=next_tok, past_key_values=past_ref, use_cache=True)
            step_tr = tr(input_ids=next_tok, past_key_values=past_tr, use_cache=True)

        assert torch.isfinite(step_tr.logits).all()
        assert torch.allclose(step_ref.logits, step_tr.logits, rtol=1e-2, atol=1e-2)

        past_ref = step_ref.past_key_values
        past_tr = step_tr.past_key_values
