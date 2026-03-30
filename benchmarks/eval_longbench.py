"""LongBench subsets with tail-context truncation (GPT-2 context limit)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmarks.eval_common import effective_max_context, greedy_generate_with_cache, truncate_context_keep_query_tail

_LONGBENCH_ZIP_PATH: str | None = None


def _longbench_zip_path() -> str:
    global _LONGBENCH_ZIP_PATH
    if _LONGBENCH_ZIP_PATH is None:
        _LONGBENCH_ZIP_PATH = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    return _LONGBENCH_ZIP_PATH


def load_longbench_subset(subset: str):
    """Rows from THUDM/LongBench `data/{subset}.jsonl` inside `data.zip` (works without legacy dataset scripts)."""
    zp = _longbench_zip_path()
    uri = f"zip://data/{subset}.jsonl::{zp}"
    return load_dataset("json", data_files=uri, split="train")


def normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def longbench_substring_hit(prediction: str, answers: list[str]) -> bool:
    pred = prediction.lower()
    pred = re.sub(r"\s+", " ", pred).strip()
    for a in answers:
        s = str(a).strip().lower()
        if not s:
            continue
        if s in pred:
            return True
    return False


def row_to_context_query(row: dict[str, Any]) -> tuple[str, str]:
    ctx = str(row.get("context", "") or "")
    inp = str(row.get("input", "") or "")
    return ctx, inp


@dataclass
class LongBenchSubsetResult:
    subset: str
    hit_rate_ref: float
    hit_rate_quant: float
    greedy_parity_rate: float
    n_evaluated: int
    truncated: bool
    effective_context_tokens: int


@dataclass
class LongBenchAggregate:
    results: list[LongBenchSubsetResult]
    hit_rate_ref_mean: float
    hit_rate_quant_mean: float
    greedy_parity_rate_mean: float
    longbench_truncated: bool
    longbench_prompt_token_cap: int


def _eval_subset(
    model_ref: GPT2LMHeadModel,
    model_quant: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    subset: str,
    max_samples: int,
    prompt_cap: int,
    max_new_tokens: int,
) -> LongBenchSubsetResult:
    ds = load_longbench_subset(subset)
    truncated_flag = False
    hits_r = 0
    hits_q = 0
    parity = 0
    n = 0
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        ctx, inp = row_to_context_query(row)
        answers = normalize_answers(row.get("answers"))
        if not answers:
            continue
        query_suffix = "\n" + inp
        ids_r = truncate_context_keep_query_tail(tokenizer, ctx, query_suffix, prompt_cap)
        if ids_r.shape[1] > prompt_cap:
            ids_r = ids_r[:, -prompt_cap:]
        raw_len = row.get("length")
        if raw_len is not None and int(raw_len) > prompt_cap:
            truncated_flag = True
        elif ids_r.shape[1] >= prompt_cap:
            truncated_flag = True
        ids_q = ids_r.clone()

        gen_r = greedy_generate_with_cache(model_ref, ids_r, max_new_tokens, device)
        gen_q = greedy_generate_with_cache(model_quant, ids_q, max_new_tokens, device)
        text_r = tokenizer.decode(gen_r[0], skip_special_tokens=True)
        text_q = tokenizer.decode(gen_q[0], skip_special_tokens=True)

        if longbench_substring_hit(text_r, answers):
            hits_r += 1
        if longbench_substring_hit(text_q, answers):
            hits_q += 1
        if torch.equal(gen_r.cpu(), gen_q.cpu()):
            parity += 1
        n += 1

    def rate(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    return LongBenchSubsetResult(
        subset=subset,
        hit_rate_ref=rate(hits_r, n),
        hit_rate_quant=rate(hits_q, n),
        greedy_parity_rate=rate(parity, n),
        n_evaluated=n,
        truncated=truncated_flag,
        effective_context_tokens=prompt_cap,
    )


def run_longbench_eval(
    model_ref: GPT2LMHeadModel,
    model_quant: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    *,
    subsets: list[str],
    max_samples_per_subset: int,
    max_context_tokens: int | None,
    max_new_tokens: int,
) -> LongBenchAggregate:
    cap = effective_max_context(model_ref, max_context_tokens)
    n_pos = int(model_ref.config.n_positions)
    prompt_cap = min(cap, n_pos - max_new_tokens)
    prompt_cap = max(1, prompt_cap)

    results: list[LongBenchSubsetResult] = []
    for sub in subsets:
        r = _eval_subset(
            model_ref,
            model_quant,
            tokenizer,
            device,
            sub,
            max_samples_per_subset,
            prompt_cap,
            max_new_tokens,
        )
        results.append(r)

    if not results:
        return LongBenchAggregate(
            results=[],
            hit_rate_ref_mean=0.0,
            hit_rate_quant_mean=0.0,
            greedy_parity_rate_mean=0.0,
            longbench_truncated=False,
            longbench_prompt_token_cap=prompt_cap,
        )

    def mean_attr(attr: str) -> float:
        return sum(getattr(x, attr) for x in results) / len(results)

    any_trunc = any(x.truncated for x in results)
    return LongBenchAggregate(
        results=results,
        hit_rate_ref_mean=mean_attr("hit_rate_ref"),
        hit_rate_quant_mean=mean_attr("hit_rate_quant"),
        greedy_parity_rate_mean=mean_attr("greedy_parity_rate"),
        longbench_truncated=any_trunc,
        longbench_prompt_token_cap=prompt_cap,
    )
