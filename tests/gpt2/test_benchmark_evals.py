"""Unit tests for benchmark eval helpers (no CUDA required)."""

from types import SimpleNamespace

import torch
from transformers import GPT2Tokenizer

from benchmarks.gpt2.eval_calibration import compute_lm_ece
from benchmarks.gpt2.eval_longbench import (
    longbench_substring_hit,
    normalize_answers,
    row_to_context_query,
)
from benchmarks.gpt2.eval_passkey import build_passkey_prompt


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


class _IncrementalCalibToy(torch.nn.Module):
    """Teacher-forced steps: each (1,1) forward predicts the next token in a fixed sequence."""

    def __init__(self, vocab_size: int, full_ids: torch.Tensor):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("full_ids", full_ids.long())
        self._t = 0
        self._dummy_past = object()

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        assert input_ids.shape == (1, 1)
        assert int(input_ids[0, 0].item()) == int(self.full_ids[0, self._t].item())
        v = self.vocab_size
        logits = torch.full((1, 1, v), -30.0, device=input_ids.device, dtype=torch.float32)
        if self._t + 1 < self.full_ids.shape[1]:
            tgt = int(self.full_ids[0, self._t + 1].item())
            if 0 <= tgt < v:
                logits[0, 0, tgt] = 20.0
        self._t += 1
        return SimpleNamespace(logits=logits, past_key_values=self._dummy_past)


def test_compute_lm_ece_near_zero_on_perfect_toy():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = ["Hello world, this is a short calibration line."]
    enc = tok(texts[0], truncation=True, max_length=32, return_tensors="pt", add_special_tokens=True)
    full_ids = enc["input_ids"]
    model = _IncrementalCalibToy(tok.vocab_size, full_ids)
    model.eval()
    ece, n = compute_lm_ece(model, tok, texts, torch.device("cpu"), max_length=32, num_bins=10)
    assert n > 0
    assert ece < 0.05
