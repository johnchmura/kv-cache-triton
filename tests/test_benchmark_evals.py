"""Unit tests for benchmark eval helpers (no CUDA required)."""

from types import SimpleNamespace

import torch
from transformers import GPT2Tokenizer

from benchmarks.eval_calibration import compute_lm_ece
from benchmarks.eval_longbench import (
    longbench_substring_hit,
    normalize_answers,
    row_to_context_query,
)
from benchmarks.eval_passkey import build_passkey_prompt


def test_normalize_answers_json_list():
    assert normalize_answers('["x", "y"]') == ["x", "y"]


def test_normalize_answers_plain_string():
    assert normalize_answers("hello") == ["hello"]


def test_longbench_substring_hit():
    assert longbench_substring_hit("The answer is Paris today.", ["paris"])
    assert not longbench_substring_hit("no match", ["paris"])


def test_row_to_context_query():
    ctx, inp = row_to_context_query({"context": "c", "input": "q"})
    assert ctx == "c" and inp == "q"


def test_build_passkey_prompt_shape():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids, gold = build_passkey_prompt(tok, "abc12345", 128, "mid", torch.device("cpu"))
    assert gold == "abc12345"
    assert ids.shape[0] == 1
    assert ids.shape[1] <= 1024


class _ToyLM(torch.nn.Module):
    """Next-token logits favor the true target (low ECE)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        b, t = input_ids.shape
        v = self.vocab_size
        logits = torch.full((b, t, v), -30.0, dtype=torch.float32)
        for i in range(t - 1):
            tgt = int(input_ids[0, i + 1].item())
            if 0 <= tgt < v:
                logits[0, i, tgt] = 20.0
        return SimpleNamespace(logits=logits)


def test_compute_lm_ece_near_zero_on_perfect_toy():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _ToyLM(tok.vocab_size)
    model.eval()
    texts = ["Hello world, this is a short calibration line."]
    ece, n = compute_lm_ece(model, tok, texts, torch.device("cpu"), max_length=32, num_bins=10)
    assert n > 0
    assert ece < 0.05
