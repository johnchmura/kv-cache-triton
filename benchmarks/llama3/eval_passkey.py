"""Simple passkey retrieval harness for Llama 3.1.

Generates a haystack text, buries a ``PASSKEY: <random>`` line at a chosen
position, and asks the model to regurgitate the passkey. Measures exact
substring match between the model's greedy continuation and the passkey.
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict, dataclass
from typing import Literal

from datasets import load_dataset

import torch
from transformers import AutoTokenizer, DynamicCache

from benchmarks.llama3.kv_cache_metrics import kv_cache_nbytes


@dataclass
class PasskeyResult:
    n_samples: int
    exact_match_rate_ref: float
    exact_match_rate_quant: float
    greedy_parity_rate: float
    avg_context_tokens: float
    avg_quant_kv_mib: float


@dataclass
class PasskeySweepPoint:
    haystack_tokens: int
    seed: int
    group_size: int
    passkey_len: int
    max_new_tokens: int
    result: PasskeyResult


@dataclass
class PasskeySample:
    sample_id: str
    prompt: str
    answers: list[str]
    scoring: Literal["match_all", "match_any"]
    n_in: int


@dataclass
class PasskeyBranchSampleResult:
    sample_id: str
    answer_text: str
    hit: bool
    n_in: int
    kv_mib: float | None


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


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _score_ruler(decoded: str, answers: list[str], scoring: Literal["match_all", "match_any"]) -> bool:
    pred = _normalize(decoded)
    gold = [_normalize(a) for a in (answers or []) if _normalize(a)]
    if not gold:
        return False
    if scoring == "match_all":
        return all(a in pred for a in gold)
    return any(a in pred for a in gold)


def load_ruler_niah_samples(
    tokenizer: AutoTokenizer,
    *,
    context_tokens: int,
    n_samples: int,
    seed: int,
    variant: str = "plain",
    split: str = "validation",
    dataset_id: str = "tonychenxyz/ruler-full",
) -> list[PasskeySample]:
    """Load RULER NIAH examples at a given context length.

    Uses `tonychenxyz/ruler-full` where rows include a ready-to-use `prompt` plus
    `extra_info` containing ground-truth answers and scoring.
    """
    # Stream to avoid downloading the full dataset for a small sample.
    ds = load_dataset(dataset_id, variant, split=split, streaming=True)
    rows = []
    suffix = f"_{int(context_tokens)}"
    for row in ds:
        cat = str(row.get("category", "") or "")
        if "niah_" not in cat:
            continue
        if suffix not in cat:
            continue
        prompt = str(row.get("prompt", "") or "")
        extra = row.get("extra_info") or {}
        answers = (
            extra.get("ground_truth")
            or extra.get("answers")
            or extra.get("answer")
            or extra.get("gold")
            or []
        )
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a) for a in answers if str(a).strip()]
        scoring_func = str(extra.get("scoring_function", "") or extra.get("scoring_func", "") or "")
        scoring: Literal["match_all", "match_any"] = "match_any"
        if "match_all" in scoring_func:
            scoring = "match_all"
        elif "match_part" in scoring_func:
            scoring = "match_any"
        if not prompt or not answers:
            continue
        n_in = len(tokenizer.encode(prompt, add_special_tokens=True))
        rows.append((cat, prompt, answers, scoring, n_in))
        # Stop early once we have enough candidates to shuffle.
        if len(rows) >= max(50, int(n_samples) * 10):
            break
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[PasskeySample] = []
    for i, (cat, prompt, answers, scoring, n_in) in enumerate(rows[: max(0, int(n_samples))]):
        out.append(
            PasskeySample(
                sample_id=f"{cat}#{i}",
                prompt=prompt,
                answers=answers,
                scoring=scoring,
                n_in=int(n_in),
            )
        )
    return out


def run_passkey_branch(
    model,
    tokenizer: AutoTokenizer,
    device: torch.device,
    samples: list[PasskeySample],
    *,
    cache_kind: Literal["dynamic", "quant"],
    group_size: int,
    max_new_tokens: int,
    quant_cache_cls=None,
) -> list[PasskeyBranchSampleResult]:
    out: list[PasskeyBranchSampleResult] = []
    for s in samples:
        if cache_kind == "dynamic":
            cache = DynamicCache()
        else:
            if quant_cache_cls is None:
                from models.llama3.kv_cache import QuantizedKVCache as _QuantizedKVCache

                quant_cache_cls = _QuantizedKVCache
            cache = quant_cache_cls(group_size=group_size)
        text, n_in = _greedy_continue(model, tokenizer, s.prompt, device, max_new_tokens, cache)
        kv_mib = None
        if cache_kind == "quant":
            kv_mib = kv_cache_nbytes(cache) / 1024**2
        del cache
        torch.cuda.empty_cache()
        hit = _score_ruler(text, s.answers, s.scoring)
        out.append(
            PasskeyBranchSampleResult(
                sample_id=s.sample_id,
                answer_text=text,
                hit=bool(hit),
                n_in=int(n_in),
                kv_mib=kv_mib,
            )
        )
    return out


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
    quant_cache_cls=None,
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

        if quant_cache_cls is None:
            from models.llama3.kv_cache import QuantizedKVCache as _QuantizedKVCache

            quant_cache_cls = _QuantizedKVCache
        cache_q = quant_cache_cls(group_size=group_size)
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


def run_passkey_sweep(
    model_ref,
    model_quant,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    group_size: int,
    haystack_tokens_sweep: list[int],
    n_samples: int = 8,
    passkey_len: int = 8,
    max_new_tokens: int = 16,
    seed: int = 0,
    seed_stride: int = 10_000,
) -> list[PasskeySweepPoint]:
    points: list[PasskeySweepPoint] = []
    for i, haystack_tokens in enumerate(haystack_tokens_sweep):
        point_seed = seed + i * seed_stride
        res = run_passkey_eval(
            model_ref,
            model_quant,
            tokenizer,
            device,
            group_size=group_size,
            n_samples=n_samples,
            passkey_len=passkey_len,
            haystack_tokens=haystack_tokens,
            max_new_tokens=max_new_tokens,
            seed=point_seed,
        )
        points.append(
            PasskeySweepPoint(
                haystack_tokens=int(haystack_tokens),
                seed=int(point_seed),
                group_size=int(group_size),
                passkey_len=int(passkey_len),
                max_new_tokens=int(max_new_tokens),
                result=res,
            )
        )
    return points


def passkey_sweep_points_to_dict(points: list[PasskeySweepPoint]) -> list[dict]:
    out: list[dict] = []
    for p in points:
        row = asdict(p)
        row.update(row.pop("result"))
        out.append(row)
    return out
