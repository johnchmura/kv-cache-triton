"""LongBench substring-hit eval for Llama 3 (tail-keep truncation).

Downloads ``THUDM/LongBench`` subsets on first run (requires network and HF
cache). For each example, truncates the context to fit
``max_context_tokens`` with tail-keep (newest context kept, plus full query
suffix), greedy-decodes ``max_new_tokens``, and scores a substring hit against
the first reference answer. Not comparable to official LongBench scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DynamicCache

from benchmarks.llama3.kv_cache_metrics import kv_cache_nbytes
from models.llama3.kv_cache import QuantizedKVCache


@dataclass
class SubsetResult:
    subset: str
    n_samples: int
    hit_rate_ref: float
    hit_rate_quant: float
    greedy_parity_rate: float
    avg_context_tokens: float
    avg_quant_kv_mib: float


@dataclass
class LongBenchResult:
    subsets: list[str]
    max_context_tokens: int
    max_new_tokens: int
    results: list[SubsetResult] = field(default_factory=list)
    hit_rate_ref_mean: float = 0.0
    hit_rate_quant_mean: float = 0.0
    greedy_parity_rate_mean: float = 0.0


def _tail_truncate(
    tokenizer: AutoTokenizer,
    context: str,
    query_suffix: str,
    max_tokens: int,
) -> tuple[str, int]:
    suffix_ids = tokenizer.encode(query_suffix, add_special_tokens=False)
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    budget = max_tokens - len(suffix_ids) - 8
    if budget <= 0:
        return tokenizer.decode(suffix_ids[-max_tokens:], skip_special_tokens=True), max_tokens
    if len(ctx_ids) > budget:
        ctx_ids = ctx_ids[-budget:]
    combined_ids = ctx_ids + suffix_ids
    return tokenizer.decode(combined_ids, skip_special_tokens=True), len(combined_ids)


def _greedy(model, tokenizer, prompt: str, device, max_new_tokens: int, cache) -> tuple[str, int]:
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
    return tokenizer.decode(tokens, skip_special_tokens=True), n_in


def _load_subset(subset: str, max_samples: int):
    ds = load_dataset("THUDM/LongBench", subset, split="test")
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds


def run_longbench_eval(
    model_ref,
    model_quant,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    group_size: int,
    subsets: list[str],
    max_samples_per_subset: int,
    max_context_tokens: int,
    max_new_tokens: int,
) -> LongBenchResult:
    result = LongBenchResult(
        subsets=subsets,
        max_context_tokens=max_context_tokens,
        max_new_tokens=max_new_tokens,
    )
    if max_samples_per_subset <= 0 or not subsets:
        return result

    all_ref = []
    all_q = []
    all_parity = []

    for subset in subsets:
        ds = _load_subset(subset, max_samples_per_subset)
        hits_ref = 0
        hits_q = 0
        parity = 0
        total_ctx = 0
        total_kv = 0.0
        for row in ds:
            context = row.get("context", "") or ""
            question = row.get("input", "") or ""
            answers = row.get("answers") or [""]
            target = answers[0] if answers else ""
            query_suffix = f"\n\nQuestion: {question}\nAnswer:"
            prompt, n_in = _tail_truncate(tokenizer, context, query_suffix, max_context_tokens)
            total_ctx += n_in

            cache_ref = DynamicCache()
            ans_ref, _ = _greedy(model_ref, tokenizer, prompt, device, max_new_tokens, cache_ref)
            del cache_ref
            torch.cuda.empty_cache()

            cache_q = QuantizedKVCache(group_size=group_size)
            ans_q, _ = _greedy(model_quant, tokenizer, prompt, device, max_new_tokens, cache_q)
            total_kv += kv_cache_nbytes(cache_q) / 1024**2
            del cache_q
            torch.cuda.empty_cache()

            if target and target.lower() in ans_ref.lower():
                hits_ref += 1
            if target and target.lower() in ans_q.lower():
                hits_q += 1
            if ans_ref.strip() == ans_q.strip():
                parity += 1

        n = max(1, len(ds))
        sr = SubsetResult(
            subset=subset,
            n_samples=len(ds),
            hit_rate_ref=hits_ref / n,
            hit_rate_quant=hits_q / n,
            greedy_parity_rate=parity / n,
            avg_context_tokens=total_ctx / n,
            avg_quant_kv_mib=total_kv / n,
        )
        result.results.append(sr)
        all_ref.append(sr.hit_rate_ref)
        all_q.append(sr.hit_rate_quant)
        all_parity.append(sr.greedy_parity_rate)

    if all_ref:
        result.hit_rate_ref_mean = sum(all_ref) / len(all_ref)
        result.hit_rate_quant_mean = sum(all_q) / len(all_q)
        result.greedy_parity_rate_mean = sum(all_parity) / len(all_parity)
    return result
