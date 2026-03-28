"""When GPT-2 uses Triton vs Hugging Face attention.

Triton now runs for *all* self-attention on CUDA (both prefill and decode),
unless eager attention is combined with reorder_and_upcast_attn.
"""

from unittest.mock import patch

import pytest
import torch
from transformers import GPT2LMHeadModel

from models import gpt2_triton
from models.gpt2_triton import replace_gpt2_attention_with_triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_prefill_invokes_triton_once_per_layer():
    calls = []
    real = gpt2_triton.attention_forward

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
    replace_gpt2_attention_with_triton(model)
    input_ids = torch.randint(0, 50257, (1, 16), device="cuda", dtype=torch.long)

    with patch.object(gpt2_triton, "attention_forward", side_effect=wrapped):
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=True)

    assert len(calls) == model.config.n_layer


@cuda
def test_decode_invokes_triton_once_per_layer():
    calls = []
    real = gpt2_triton.attention_forward

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    torch.manual_seed(1)
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
    replace_gpt2_attention_with_triton(model)
    input_ids = torch.randint(0, 50257, (1, 16), device="cuda", dtype=torch.long)

    with patch.object(gpt2_triton, "attention_forward", side_effect=wrapped):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        calls.clear()
        next_tok = torch.randint(0, 50257, (1, 1), device="cuda", dtype=torch.long)
        with torch.no_grad():
            model(input_ids=next_tok, past_key_values=out.past_key_values, use_cache=True)

    assert len(calls) == model.config.n_layer


@cuda
def test_decode_skips_triton_when_reorder_and_upcast_eager():
    calls = []
    real = gpt2_triton.attention_forward

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    torch.manual_seed(2)
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").cuda().half().eval()
    model.config.reorder_and_upcast_attn = True
    replace_gpt2_attention_with_triton(model)
    input_ids = torch.randint(0, 50257, (1, 16), device="cuda", dtype=torch.long)

    with patch.object(gpt2_triton, "attention_forward", side_effect=wrapped):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        calls.clear()
        next_tok = torch.randint(0, 50257, (1, 1), device="cuda", dtype=torch.long)
        with torch.no_grad():
            model(input_ids=next_tok, past_key_values=out.past_key_values, use_cache=True)

    assert len(calls) == 0
