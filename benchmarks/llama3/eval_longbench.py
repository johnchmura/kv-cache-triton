"""LongBench substring-hit eval for Llama 3 (tail-keep truncation).

Downloads ``THUDM/LongBench`` subsets on first run (requires network and HF
cache). For each example, truncates the context to fit
``max_context_tokens`` with tail-keep (newest context kept, plus full query
suffix), greedy-decodes ``max_new_tokens``, and scores a substring hit against
the first reference answer. Not comparable to official LongBench scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
_LONGBENCH_ZIP_PATH: str | None = None


def _longbench_zip_path() -> str:
    global _LONGBENCH_ZIP_PATH
    if _LONGBENCH_ZIP_PATH is None:
        _LONGBENCH_ZIP_PATH = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    return _LONGBENCH_ZIP_PATH


def _load_longbench_subset_json(subset: str, max_samples: int):
    """Rows from THUDM/LongBench `data/{subset}.jsonl` inside `data.zip` (works without legacy dataset scripts)."""
    zp = _longbench_zip_path()
    uri = f"zip://data/{subset}.jsonl::{zp}"
    ds = load_dataset("json", data_files=uri, split="train")
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds
from transformers import AutoTokenizer, DynamicCache

from benchmarks.llama3.kv_cache_metrics import kv_cache_nbytes


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


@dataclass
class LongBenchSweepPoint:
    max_context_tokens: int
    max_new_tokens: int
    group_size: int
    max_samples_per_subset: int
    subsets: list[str]
    hit_rate_ref_mean_subset: float
    hit_rate_quant_mean_subset: float
    greedy_parity_rate_mean_subset: float
    hit_rate_ref_mean_sample: float
    hit_rate_quant_mean_sample: float
    greedy_parity_rate_mean_sample: float
    avg_context_tokens_mean_sample: float
    avg_quant_kv_mib_mean_sample: float
    by_subset: dict[str, dict]


@dataclass
class LongBenchExample:
    subset: str
    row_idx: int
    prompt: str
    target: str
    n_in: int


@dataclass
class LongBenchBranchExampleResult:
    subset: str
    row_idx: int
    answer_text: str
    hit: bool
    n_in: int
    kv_mib: float | None


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
    return _load_longbench_subset_json(subset, max_samples)


def build_longbench_examples(
    tokenizer: AutoTokenizer,
    *,
    subset: str,
    max_samples: int,
    max_context_tokens: int,
) -> list[LongBenchExample]:
    ds = _load_subset(subset, max_samples)
    out: list[LongBenchExample] = []
    for i, row in enumerate(ds):
        context = row.get("context", "") or ""
        question = row.get("input", "") or ""
        answers = row.get("answers") or [""]
        target = answers[0] if answers else ""
        query_suffix = f"\n\nQuestion: {question}\nAnswer:"
        prompt, n_in = _tail_truncate(tokenizer, context, query_suffix, max_context_tokens)
        out.append(
            LongBenchExample(
                subset=subset,
                row_idx=int(i),
                prompt=str(prompt),
                target=str(target),
                n_in=int(n_in),
            )
        )
    return out


def run_longbench_branch(
    model,
    tokenizer: AutoTokenizer,
    device: torch.device,
    examples: list[LongBenchExample],
    *,
    cache_kind: Literal["dynamic", "quant"],
    group_size: int,
    max_new_tokens: int,
    quant_cache_cls=None,
) -> list[LongBenchBranchExampleResult]:
    out: list[LongBenchBranchExampleResult] = []
    for ex in examples:
        if cache_kind == "dynamic":
            cache = DynamicCache()
        else:
            if quant_cache_cls is None:
                from models.llama3.kv_cache import QuantizedKVCache as _QuantizedKVCache

                quant_cache_cls = _QuantizedKVCache
            cache = quant_cache_cls(group_size=group_size)
        ans, n_in = _greedy(model, tokenizer, ex.prompt, device, max_new_tokens, cache)
        kv_mib = None
        if cache_kind == "quant":
            kv_mib = kv_cache_nbytes(cache) / 1024**2
        del cache
        torch.cuda.empty_cache()
        hit = bool(ex.target and ex.target.lower() in (ans or "").lower())
        out.append(
            LongBenchBranchExampleResult(
                subset=ex.subset,
                row_idx=int(ex.row_idx),
                answer_text=str(ans),
                hit=hit,
                n_in=int(n_in),
                kv_mib=kv_mib,
            )
        )
    return out


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
    quant_cache_cls=None,
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

            if quant_cache_cls is None:
                from models.llama3.kv_cache import QuantizedKVCache as _QuantizedKVCache

                quant_cache_cls = _QuantizedKVCache
            cache_q = quant_cache_cls(group_size=group_size)
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


def run_longbench_sweep(
    model_ref,
    model_quant,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    group_size: int,
    subsets: list[str],
    max_samples_per_subset: int,
    max_context_tokens_sweep: list[int],
    max_new_tokens: int,
) -> list[LongBenchSweepPoint]:
    points: list[LongBenchSweepPoint] = []
    for max_ctx in max_context_tokens_sweep:
        lb = run_longbench_eval(
            model_ref,
            model_quant,
            tokenizer,
            device,
            group_size=group_size,
            subsets=subsets,
            max_samples_per_subset=max_samples_per_subset,
            max_context_tokens=int(max_ctx),
            max_new_tokens=int(max_new_tokens),
        )
        by_subset = {r.subset: asdict(r) for r in lb.results}
        total_n = sum(int(r.n_samples) for r in lb.results) or 1
        hit_ref_sample = sum(float(r.hit_rate_ref) * int(r.n_samples) for r in lb.results) / total_n
        hit_quant_sample = sum(float(r.hit_rate_quant) * int(r.n_samples) for r in lb.results) / total_n
        parity_sample = sum(float(r.greedy_parity_rate) * int(r.n_samples) for r in lb.results) / total_n
        ctx_tok_sample = sum(float(r.avg_context_tokens) * int(r.n_samples) for r in lb.results) / total_n
        kv_mib_sample = sum(float(r.avg_quant_kv_mib) * int(r.n_samples) for r in lb.results) / total_n
        points.append(
            LongBenchSweepPoint(
                max_context_tokens=int(lb.max_context_tokens),
                max_new_tokens=int(lb.max_new_tokens),
                group_size=int(group_size),
                max_samples_per_subset=int(max_samples_per_subset),
                subsets=list(subsets),
                hit_rate_ref_mean_subset=float(lb.hit_rate_ref_mean),
                hit_rate_quant_mean_subset=float(lb.hit_rate_quant_mean),
                greedy_parity_rate_mean_subset=float(lb.greedy_parity_rate_mean),
                hit_rate_ref_mean_sample=float(hit_ref_sample),
                hit_rate_quant_mean_sample=float(hit_quant_sample),
                greedy_parity_rate_mean_sample=float(parity_sample),
                avg_context_tokens_mean_sample=float(ctx_tok_sample),
                avg_quant_kv_mib_mean_sample=float(kv_mib_sample),
                by_subset=by_subset,
            )
        )
    return points


def longbench_sweep_points_to_dict(points: list[LongBenchSweepPoint]) -> list[dict]:
    return [asdict(p) for p in points]
