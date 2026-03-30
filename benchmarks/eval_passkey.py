"""Passkey (needle) retrieval and optional RULER-style multi-key recall."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmarks.eval_common import effective_max_context, greedy_generate_with_cache

FILLER_UNIT = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
)


@dataclass
class PasskeyMetrics:
    exact_match_rate_ref: float
    exact_match_rate_quant: float
    greedy_parity_rate: float
    n_samples: int
    ruler_style: bool
    n_positions_start: int
    n_positions_mid: int
    n_positions_end: int


def _random_passkey(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_passkey_prompt(
    tokenizer: GPT2Tokenizer,
    passkey: str,
    haystack_tokens_target: int,
    position: Literal["start", "mid", "end"],
    device: torch.device,
    suffix: str = "\n\nWhat is the pass key?\nThe pass key is ",
) -> tuple[torch.Tensor, str]:
    """Returns input_ids (1, seq) and gold answer string (passkey)."""
    needle = f"The pass key is {passkey}."
    filler_ids = tokenizer(FILLER_UNIT, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if filler_ids.numel() == 0:
        filler_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)

    needle_ids = tokenizer(needle, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    def repeat_filler(n: int) -> torch.Tensor:
        if n <= 0:
            return torch.tensor([], dtype=torch.long)
        reps = (n + filler_ids.numel() - 1) // filler_ids.numel()
        cat = filler_ids.repeat(reps)[:n]
        return cat

    core_budget = haystack_tokens_target - needle_ids.numel() - suffix_ids.numel()
    if core_budget < 0:
        core_budget = 0

    if position == "start":
        before, after = needle_ids, repeat_filler(core_budget)
    elif position == "end":
        before, after = repeat_filler(core_budget), needle_ids
    else:
        b = core_budget // 2
        before = repeat_filler(b)
        after = torch.cat([needle_ids, repeat_filler(core_budget - b)], dim=0)

    body = torch.cat([before, after], dim=0)
    full = torch.cat([body, suffix_ids], dim=0).unsqueeze(0).to(device)
    return full, passkey


def build_ruler_two_key_prompt(
    tokenizer: GPT2Tokenizer,
    key_keep: str,
    key_distractor: str,
    haystack_tokens_target: int,
    position: Literal["start", "mid", "end"],
    device: torch.device,
    suffix: str = "\n\nWhich pass key was marked as primary?\nThe primary pass key is ",
) -> tuple[torch.Tensor, str]:
    block = (
        f"The secondary pass key is {key_distractor}. "
        f"The primary pass key is {key_keep}. "
        f"Ignore the secondary key for the final answer."
    )
    filler_ids = tokenizer(FILLER_UNIT, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if filler_ids.numel() == 0:
        filler_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)

    block_ids = tokenizer(block, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    def repeat_filler(n: int) -> torch.Tensor:
        if n <= 0:
            return torch.tensor([], dtype=torch.long)
        reps = (n + filler_ids.numel() - 1) // filler_ids.numel()
        return filler_ids.repeat(reps)[:n]

    core_budget = haystack_tokens_target - block_ids.numel() - suffix_ids.numel()
    if core_budget < 0:
        core_budget = 0

    if position == "start":
        body = torch.cat([block_ids, repeat_filler(core_budget)], dim=0)
    elif position == "end":
        body = torch.cat([repeat_filler(core_budget), block_ids], dim=0)
    else:
        b = core_budget // 2
        body = torch.cat(
            [repeat_filler(b), block_ids, repeat_filler(core_budget - b)],
            dim=0,
        )

    full = torch.cat([body, suffix_ids], dim=0).unsqueeze(0).to(device)
    return full, key_keep


def _decode_answer(tokenizer: GPT2Tokenizer, gen_ids: torch.Tensor) -> str:
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def _answer_matches(gold: str, decoded: str) -> bool:
    g = gold.strip().lower()
    d = decoded.strip().lower()
    if g in d:
        return True
    if d.startswith(g):
        return True
    first = d.split()[0] if d.split() else ""
    return first == g


def run_passkey_eval(
    model_ref: GPT2LMHeadModel,
    model_quant: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    *,
    seed: int,
    n_samples: int,
    passkey_len: int,
    haystack_tokens: int,
    max_context_tokens: int | None,
    ruler_style: bool,
) -> PasskeyMetrics:
    rng = random.Random(seed)
    n_pos = int(model_ref.config.n_positions)
    decode_tokens = max(8, passkey_len + 4)
    room = max(1, n_pos - decode_tokens)
    max_ctx = min(effective_max_context(model_ref, max_context_tokens), room)
    max_ctx = max(16, max_ctx)
    if max_ctx > room:
        max_ctx = room

    positions: list[Literal["start", "mid", "end"]] = ["start", "mid", "end"]
    hits_ref = 0
    hits_quant = 0
    parity = 0
    count = 0
    pos_counts = {"start": 0, "mid": 0, "end": 0}

    for i in range(n_samples):
        pos = positions[i % 3]
        pos_counts[pos] += 1
        if ruler_style:
            k1 = _random_passkey(rng, passkey_len)
            k2 = _random_passkey(rng, passkey_len)
            while k2 == k1:
                k2 = _random_passkey(rng, passkey_len)
            inp_ref, gold = build_ruler_two_key_prompt(
                tokenizer, k1, k2, haystack_tokens, pos, device
            )
        else:
            pk = _random_passkey(rng, passkey_len)
            inp_ref, gold = build_passkey_prompt(tokenizer, pk, haystack_tokens, pos, device)

        if inp_ref.shape[1] > max_ctx:
            inp_ref = inp_ref[:, -max_ctx:]
        inp_q = inp_ref.clone()

        gen_ref = greedy_generate_with_cache(model_ref, inp_ref, decode_tokens, device)
        gen_q = greedy_generate_with_cache(model_quant, inp_q, decode_tokens, device)

        dr = _decode_answer(tokenizer, gen_ref)
        dq = _decode_answer(tokenizer, gen_q)
        if _answer_matches(gold, dr):
            hits_ref += 1
        if _answer_matches(gold, dq):
            hits_quant += 1
        if torch.equal(gen_ref.cpu(), gen_q.cpu()):
            parity += 1
        count += 1

    def rate(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    return PasskeyMetrics(
        exact_match_rate_ref=rate(hits_ref, count),
        exact_match_rate_quant=rate(hits_quant, count),
        greedy_parity_rate=rate(parity, count),
        n_samples=count,
        ruler_style=ruler_style,
        n_positions_start=pos_counts["start"],
        n_positions_mid=pos_counts["mid"],
        n_positions_end=pos_counts["end"],
    )
