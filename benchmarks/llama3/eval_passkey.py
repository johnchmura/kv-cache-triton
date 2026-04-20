"""Simple passkey retrieval harness for Llama 3.1.

Generates a haystack text, buries a ``PASSKEY: <random>`` line at a chosen
position, and asks the model to regurgitate the passkey. Measures exact
substring match between the model's greedy continuation and the passkey.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, DynamicCache

from benchmarks.llama3.kv_cache_metrics import kv_cache_nbytes
from models.llama3.kv_cache import QuantizedKVCache


@dataclass
class PasskeyResult:
    n_samples: int
    exact_match_rate_ref: float
    exact_match_rate_quant: float
    greedy_parity_rate: float
    avg_context_tokens: float
    avg_quant_kv_mib: float


_FILLER = (
    "The grass is green, and the sky is blue. The sun rises in the east and sets in the west. "
    "Cats purr, dogs bark, birds sing, and rivers flow downhill to the sea. "
)


def _make_prompt(
    tokenizer: AutoTokenizer,
    passkey: str,
    haystack_tokens: int,
    position: str,
    rng: random.Random,
) -> tuple[str, int]:
    target_tokens = max(64, haystack_tokens)
    base_chunks: list[str] = []
    cur_tokens = 0
    while cur_tokens < target_tokens:
        base_chunks.append(_FILLER)
        cur_tokens += len(tokenizer.encode(_FILLER, add_special_tokens=False))
    needle = f"\nThe secret PASSKEY is: {passkey}. Remember this number.\n"
    if position == "start":
        prefix = needle + "".join(base_chunks)
    elif position == "end":
        prefix = "".join(base_chunks) + needle
    else:
        mid = len(base_chunks) // 2
        prefix = "".join(base_chunks[:mid]) + needle + "".join(base_chunks[mid:])
    query = "\n\nQuestion: What is the secret PASSKEY?\nAnswer: The PASSKEY is"
    prompt = prefix + query
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=True))


def _greedy_continue(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    cache,
) -> tuple[str, int]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    n_in = int(input_ids.shape[1])
    with torch.inference_mode():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens: list[int] = [int(next_id.item())]
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(int(next_id.item()))
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, n_in


def _random_passkey(rng: random.Random, length: int) -> str:
    alphabet = string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def run_passkey_eval(
    model_ref,
    model_quant,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    group_size: int,
    n_samples: int = 8,
    passkey_len: int = 8,
    haystack_tokens: int = 4096,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> PasskeyResult:
    if n_samples <= 0:
        return PasskeyResult(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    positions = ["start", "mid", "end"]
    rng = random.Random(seed)

    hits_ref = 0
    hits_q = 0
    parity = 0
    total_ctx = 0
    total_kv = 0.0

    for i in range(n_samples):
        passkey = _random_passkey(rng, passkey_len)
        position = positions[i % len(positions)]
        prompt, n_in = _make_prompt(tokenizer, passkey, haystack_tokens, position, rng)
        total_ctx += n_in

        cache_ref = DynamicCache()
        ans_ref, _ = _greedy_continue(
            model_ref, tokenizer, prompt, device, max_new_tokens, cache_ref
        )
        del cache_ref
        torch.cuda.empty_cache()

        cache_q = QuantizedKVCache(group_size=group_size)
        ans_q, _ = _greedy_continue(
            model_quant, tokenizer, prompt, device, max_new_tokens, cache_q
        )
        total_kv += kv_cache_nbytes(cache_q) / 1024**2
        del cache_q
        torch.cuda.empty_cache()

        if passkey in ans_ref:
            hits_ref += 1
        if passkey in ans_q:
            hits_q += 1
        if ans_ref.strip() == ans_q.strip():
            parity += 1

    return PasskeyResult(
        n_samples=n_samples,
        exact_match_rate_ref=hits_ref / n_samples,
        exact_match_rate_quant=hits_q / n_samples,
        greedy_parity_rate=parity / n_samples,
        avg_context_tokens=total_ctx / n_samples,
        avg_quant_kv_mib=total_kv / n_samples,
    )
