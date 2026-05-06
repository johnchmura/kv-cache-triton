"""Benchmark Llama 3.1 baseline (BF16 KV / stock ``DynamicCache``) vs quantized (INT4 KV).

Sweeps one or more Llama 3.1 checkpoints (default: 8B and 70B base) through a
configurable prefill/decode grid. For each model we load the weights exactly
once with ``device_map="auto"``, run the reference pass with a stock
``DynamicCache``, then swap attention modules in-place to the fused INT4 path
(``replace_llama_attention_with_quantized`` re-uses the existing q/k/v/o
parameters, no second weight copy) and run the quantized pass.

Per-length metrics (decode latency, peak VRAM summed across all visible
devices, KV tensor bytes, tokens/s) plus optional quality evals
(WikiText-2 PPL, passkey, LongBench) land under
``benchmarks/llama3/logs/<ts>/<model_slug>/...`` with a cross-model decode
latency PNG at the run root.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.llama3.eval_longbench import (
    build_longbench_examples,
    longbench_sweep_points_to_dict,
    run_longbench_branch,
    run_longbench_eval,
    run_longbench_sweep,
)
from benchmarks.llama3.eval_passkey import (
    load_ruler_niah_samples,
    passkey_sweep_points_to_dict,
    run_passkey_branch,
    run_passkey_eval,
    run_passkey_sweep,
)
from benchmarks.llama3.bench_log import BenchLogger
from benchmarks.llama3.kv_cache_metrics import kv_cache_nbytes
from models.llama3.kernel_variant import bind_quantized_kernel_variant, resolve_kernel_variant
from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.llama3_quant import replace_llama_attention_with_quantized


def _resolve_kernel_variant(choice: str) -> str:
    """Resolve ``--kernel-variant`` to ``default`` or ``sm70`` (see :func:`resolve_kernel_variant`)."""
    return resolve_kernel_variant(choice)


def _bind_kernel_variant(variant: str) -> None:
    """Rebind module-level ``QuantizedKVCache`` / ``replace_llama_attention_with_quantized``."""
    global QuantizedKVCache, replace_llama_attention_with_quantized
    QuantizedKVCache, replace_llama_attention_with_quantized = bind_quantized_kernel_variant(variant)


DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"]
DEFAULT_PREFILL_LENS = [1024, 4096, 16384]
DEFAULT_LONGBENCH_SUBSETS = ["narrativeqa", "triviaqa"]


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _num_devices() -> int:
    return max(1, torch.cuda.device_count())


def _reset_peak_all() -> None:
    for d in range(_num_devices()):
        torch.cuda.reset_peak_memory_stats(d)


def _peak_mib_all() -> tuple[float, list[float]]:
    per = [torch.cuda.max_memory_allocated(d) / 1024**2 for d in range(_num_devices())]
    return float(sum(per)), per


def _sync_all() -> None:
    for d in range(_num_devices()):
        torch.cuda.synchronize(d)


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda", 0)


def _load_llama(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    return model


def _make_cache(quantized: bool, group_size: int):
    return QuantizedKVCache(group_size=group_size) if quantized else DynamicCache()


def _benchmark_decode(
    model,
    prefill_len: int,
    num_decode_steps: int,
    warmup: int,
    seed: int,
    quantized: bool,
    group_size: int,
    vocab_size: int,
) -> dict:
    torch.manual_seed(seed)
    in_dev = _input_device(model)
    input_ids = torch.randint(0, vocab_size, (1, prefill_len), device=in_dev, dtype=torch.long)

    def _run_once(timed: bool) -> dict:
        if timed:
            _reset_peak_all()
        _sync_all()
        t0 = time.perf_counter()
        cache = _make_cache(quantized, group_size)
        with torch.inference_mode():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        _sync_all()
        t1 = time.perf_counter()
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        t2 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_decode_steps):
                out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
                next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        _sync_all()
        t3 = time.perf_counter()
        peak_total, peak_per = _peak_mib_all()
        res = {
            "prefill_ms": (t1 - t0) * 1000.0,
            "decode_ms": (t3 - t2) * 1000.0,
            "peak_cuda_mib": peak_total,
            "peak_cuda_mib_per_device": peak_per,
            "kv_bytes": kv_cache_nbytes(cache),
        }
        del cache
        gc.collect()
        torch.cuda.empty_cache()
        return res

    for _ in range(max(1, warmup)):
        _run_once(timed=False)
    return _run_once(timed=True)


def _compute_ppl(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    max_batches: int,
    quantized: bool,
    group_size: int,
) -> float:
    model.eval()
    in_dev = _input_device(model)
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    for text in texts:
        if n_batches >= max_batches:
            break
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(in_dev)
        if input_ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            cache = _make_cache(quantized, group_size)
            out = model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            loss = out.loss.float().item()
            valid = int(input_ids.shape[1]) - 1
            if math.isfinite(loss) and valid > 0:
                total_nll += loss * valid
                total_tokens += valid
            del cache
            torch.cuda.empty_cache()
        n_batches += 1
    if total_tokens == 0:
        return float("nan")
    return float(math.exp(total_nll / total_tokens))


def _load_wikitext(max_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [row["text"] for row in ds if row.get("text") and row["text"].strip()]
    return texts[:max_samples]


def _plot_bars(path: Path, title: str, ylabel: str, labels: list[str], values: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(labels)))
    ax.bar(x, values, width=0.55, color=["#4477AA", "#44AA99", "#DDAA33", "#AA4499"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cross_length(path: Path, title: str, ylabel: str, rows: list[tuple[int, float, float]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[0])
    xs = [r[0] for r in rows]
    ref = [r[1] for r in rows]
    qnt = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ref, marker="o", label="reference (BF16 KV)", color="#4477AA")
    ax.plot(xs, qnt, marker="s", label="quantized (INT4 KV)", color="#44AA99")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefill length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cross_models(
    path: Path,
    title: str,
    ylabel: str,
    per_model: dict[str, list[tuple[int, float, float]]],
) -> None:
    """Overlay multiple models (ref + quant lines each) on one plot."""
    if not per_model:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette_ref = ["#4477AA", "#AA4499", "#332288"]
    palette_quant = ["#44AA99", "#DDAA33", "#117733"]
    for i, (model_slug, rows) in enumerate(per_model.items()):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r[0])
        xs = [r[0] for r in rows]
        ref = [r[1] for r in rows]
        qnt = [r[2] for r in rows]
        c_ref = palette_ref[i % len(palette_ref)]
        c_quant = palette_quant[i % len(palette_quant)]
        ax.plot(xs, ref, marker="o", linestyle="--", label=f"{model_slug} BF16", color=c_ref)
        ax.plot(xs, qnt, marker="s", linestyle="-", label=f"{model_slug} INT4", color=c_quant)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefill length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cross_length_multi(
    path: Path,
    title: str,
    ylabel: str,
    rows: list[dict[str, Any]],
    *,
    x_name: str,
    y_name: str,
    ref_key: str,
    quant_key: str,
) -> None:
    pts = []
    for r in rows:
        if ref_key not in r or quant_key not in r:
            continue
        x = r.get(x_name)
        if x is None:
            x = r.get("x_value")
        try:
            x = int(x)
            a = float(r[ref_key])
            b = float(r[quant_key])
        except Exception:
            continue
        pts.append((x, a, b))
    _plot_cross_length(path, title, ylabel, pts)


def _write_length_artifacts(out_dir: Path, metrics: dict[str, Any]) -> None:
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, (dict, list)):
                continue
            if v is None:
                continue
            w.writerow([k, v])

    _plot_bars(
        out_dir / "decode_latency.png",
        "Decode latency (lower is better)",
        "ms / token",
        ["reference", "quantized"],
        [metrics["ref_decode_ms_per_token"], metrics["quant_decode_ms_per_token"]],
    )
    _plot_bars(
        out_dir / "peak_vram.png",
        "Peak CUDA memory during decode (sum across devices)",
        "MiB (max allocated)",
        ["reference", "quantized"],
        [metrics["ref_peak_cuda_mib"], metrics["quant_peak_cuda_mib"]],
    )
    _plot_bars(
        out_dir / "kv_cache_storage_mib.png",
        "KV cache tensor storage",
        "MiB",
        ["reference", "quantized"],
        [metrics["ref_kv_cache_mib"], metrics["quant_kv_cache_mib"]],
    )


def run_one_length(
    model,
    quantized: bool,
    prefill_len: int,
    args: argparse.Namespace,
    vocab_size: int,
) -> dict[str, float]:
    seed = args.seed + (3 if quantized else 0)
    return _benchmark_decode(
        model,
        prefill_len=prefill_len,
        num_decode_steps=args.num_decode_steps,
        warmup=args.warmup,
        seed=seed,
        quantized=quantized,
        group_size=args.group_size,
        vocab_size=vocab_size,
    )


def _combine_length_metrics(
    prefill_len: int,
    num_decode_steps: int,
    ref: dict,
    qnt: dict,
) -> dict[str, Any]:
    return {
        "prefill_len": prefill_len,
        "num_decode_steps": num_decode_steps,
        "ref_prefill_ms": ref["prefill_ms"],
        "ref_decode_total_ms": ref["decode_ms"],
        "ref_decode_ms_per_token": ref["decode_ms"] / num_decode_steps,
        "ref_tokens_per_s": num_decode_steps / (ref["decode_ms"] / 1000.0),
        "ref_peak_cuda_mib": ref["peak_cuda_mib"],
        "ref_peak_cuda_mib_per_device": ref["peak_cuda_mib_per_device"],
        "ref_kv_cache_bytes": ref["kv_bytes"],
        "ref_kv_cache_mib": ref["kv_bytes"] / 1024**2,
        "quant_prefill_ms": qnt["prefill_ms"],
        "quant_decode_total_ms": qnt["decode_ms"],
        "quant_decode_ms_per_token": qnt["decode_ms"] / num_decode_steps,
        "quant_tokens_per_s": num_decode_steps / (qnt["decode_ms"] / 1000.0),
        "quant_peak_cuda_mib": qnt["peak_cuda_mib"],
        "quant_peak_cuda_mib_per_device": qnt["peak_cuda_mib_per_device"],
        "quant_kv_cache_bytes": qnt["kv_bytes"],
        "quant_kv_cache_mib": qnt["kv_bytes"] / 1024**2,
    }


def run_one_model(
    model_id: str,
    args: argparse.Namespace,
    out_root: Path,
    ts: str,
    cuda_desc: str,
    longbench_subsets: list[str],
) -> dict[str, Any]:
    """Load one checkpoint, run reference then in-place quantized, unload.

    Both paths share the same weight storage: we swap attention modules in
    place with :func:`replace_llama_attention_with_quantized` after the
    reference pass, so peak memory stays at ~model_size + KV without holding
    two copies (critical for 70B on 8x32GB).
    """
    model_slug = _model_slug(model_id)
    model_dir = out_root / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = BenchLogger(model_dir)

    print(f"[info] === model: {model_id} ===")
    print(f"[info] loading tokenizer {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"[info] loading model (BF16, device_map=auto)")
    t_load = time.perf_counter()
    model = _load_llama(model_id)
    load_s = time.perf_counter() - t_load
    print(f"[info] model loaded in {load_s:.1f}s")

    vocab_size = model.config.vocab_size
    texts = _load_wikitext(args.ppl_samples) if args.ppl_samples > 0 else []

    model_summary: dict[str, Any] = {
        "model": model_id,
        "model_slug": model_slug,
        "run_timestamp": ts,
        "cuda_description": cuda_desc,
        "group_size": args.group_size,
        "num_decode_steps": args.num_decode_steps,
        "load_seconds": load_s,
        "prefill_lens": list(args.prefill_lens),
    }
    logger.write_config(model_summary)

    # Per-model progress tracker (one line with ETA).
    passkey_sweep_tokens = list(getattr(args, "passkey_haystack_sweep", []) or [])
    longbench_ctx_sweep = list(getattr(args, "longbench_context_sweep", []) or [])
    ppl_sweep_lens = list(getattr(args, "ppl_max_seq_len_sweep", []) or [])
    if not ppl_sweep_lens and args.ppl_max_seq_len:
        ppl_sweep_lens = [int(args.ppl_max_seq_len)]

    n_prefill = len(list(args.prefill_lens))
    n_ppl = len(ppl_sweep_lens) if args.ppl_samples > 0 else 0
    n_passkey = len(passkey_sweep_tokens) if args.passkey_samples > 0 else 0
    n_longbench = (len(longbench_ctx_sweep) * len(longbench_subsets)) if (longbench_subsets and args.longbench_max_samples > 0) else 0
    total_steps = 2 * n_prefill + 2 * n_ppl + 2 * n_passkey + 2 * n_longbench
    done_steps = 0
    t_start = time.perf_counter()
    last_print_t = 0.0

    def _fmt_mmss(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60.0)
        s = int(seconds - 60.0 * m)
        return f"{m:02d}:{s:02d}"

    def progress(stage: str, *, force: bool = False) -> None:
        nonlocal last_print_t
        now = time.perf_counter()
        if not force and (now - last_print_t) < 1.0:
            return
        last_print_t = now
        if total_steps <= 0:
            return
        elapsed = max(1e-6, now - t_start)
        rate = done_steps / elapsed
        remaining = max(0, total_steps - done_steps)
        eta = remaining / rate if rate > 0 else float("inf")
        pct = 100.0 * float(done_steps) / float(total_steps)
        eta_s = _fmt_mmss(eta) if eta != float("inf") else "--:--"
        print(f"[progress] {model_slug} {done_steps}/{total_steps} ({pct:.1f}%) ETA={eta_s} stage={stage}")

    ref_runs: dict[int, dict] = {}

    print("[info] --- reference pass (stock DynamicCache, BF16 KV) ---")
    ppl_ref = None
    ppl_sweep_ref: list[dict[str, Any]] = []
    if texts and ppl_sweep_lens:
        print(f"[info] computing reference PPL sweep at max_seq_len={ppl_sweep_lens}")
        for max_len in ppl_sweep_lens:
            v = _compute_ppl(
                model, tokenizer, texts, int(max_len),
                max_batches=args.ppl_samples, quantized=False, group_size=args.group_size,
            )
            ppl_sweep_ref.append({"max_seq_len": int(max_len), "ppl": float(v)})
            logger.log(
                {
                    "run_timestamp": ts,
                    "model": model_id,
                    "model_slug": model_slug,
                    "branch": "reference",
                    "metric_family": "ppl_sweep",
                    "x_name": "max_seq_len",
                    "x_value": int(max_len),
                    "ppl": float(v),
                }
            )
            done_steps += 1
            progress(f"ppl_ref max_len={max_len}")
        ppl_ref = ppl_sweep_ref[-1]["ppl"]

    for L in args.prefill_lens:
        print(f"[info] [ref]  benchmarking prefill_len={L}")
        try:
            ref_runs[L] = run_one_length(model, quantized=False, prefill_len=L, args=args, vocab_size=vocab_size)
        except Exception as e:
            print(f"[error] [ref]  prefill_len={L} failed: {e}")
            ref_runs[L] = {"error": str(e)}
            done_steps += 1
            progress(f"decode_ref prefill_len={L} error", force=True)
        else:
            done_steps += 1
            progress(f"decode_ref prefill_len={L}")

    # Retrieval evals on the true baseline attention path (pre-swap).
    passkey_ref_by_x: dict[int, dict[str, Any]] = {}
    longbench_ref_by_x: dict[int, dict[str, Any]] = {}

    if args.passkey_samples > 0 and passkey_sweep_tokens:
        print(f"[info] RULER NIAH passkey sweep (baseline) ctx_tokens={passkey_sweep_tokens} x {args.passkey_samples}")
        try:
            for ctx_tokens in passkey_sweep_tokens:
                samples = load_ruler_niah_samples(
                    tokenizer,
                    context_tokens=int(ctx_tokens),
                    n_samples=int(args.passkey_samples),
                    seed=int(args.seed + 7),
                )
                ref_res = run_passkey_branch(
                    model,
                    tokenizer,
                    _input_device(model),
                    samples,
                    cache_kind="dynamic",
                    group_size=args.group_size,
                    max_new_tokens=16,
                    quant_cache_cls=QuantizedKVCache,
                )
                passkey_ref_by_x[int(ctx_tokens)] = {
                    "samples": samples,
                    "results": ref_res,
                }
                done_steps += 1
                progress(f"passkey_ref ctx_tokens={ctx_tokens}")
        except Exception as e:
            print(f"[error] passkey baseline sweep failed: {e}")
            done_steps += 1
            progress("passkey_ref error", force=True)

    if longbench_subsets and args.longbench_max_samples > 0 and longbench_ctx_sweep:
        print(f"[info] longbench sweep (baseline) ctx_tokens={longbench_ctx_sweep} subsets={longbench_subsets}")
        try:
            for max_ctx in longbench_ctx_sweep:
                per_subset_examples: dict[str, list[Any]] = {}
                for subset in longbench_subsets:
                    per_subset_examples[subset] = build_longbench_examples(
                        tokenizer,
                        subset=subset,
                        max_samples=int(args.longbench_max_samples),
                        max_context_tokens=int(max_ctx),
                    )
                # Run baseline branch over all examples.
                per_subset_ref: dict[str, list[Any]] = {}
                for subset, exs in per_subset_examples.items():
                    per_subset_ref[subset] = run_longbench_branch(
                        model,
                        tokenizer,
                        _input_device(model),
                        exs,
                        cache_kind="dynamic",
                        group_size=args.group_size,
                        max_new_tokens=int(args.longbench_max_new_tokens),
                        quant_cache_cls=QuantizedKVCache,
                    )
                    done_steps += 1
                    progress(f"longbench_ref ctx={max_ctx} subset={subset}")
                longbench_ref_by_x[int(max_ctx)] = {
                    "examples": per_subset_examples,
                    "results": per_subset_ref,
                }
        except Exception as e:
            print(f"[error] longbench baseline sweep failed: {e}")
            done_steps += 1
            progress("longbench_ref error", force=True)

    print("[info] --- swapping attention modules to quantized INT4 path ---")
    replace_llama_attention_with_quantized(model, group_size=args.group_size)

    ppl_q = None
    ppl_sweep_quant: list[dict[str, Any]] = []
    if texts and ppl_sweep_lens:
        print(f"[info] computing quantized PPL sweep")
        for max_len in ppl_sweep_lens:
            v = _compute_ppl(
                model, tokenizer, texts, int(max_len),
                max_batches=args.ppl_samples, quantized=True, group_size=args.group_size,
            )
            ppl_sweep_quant.append({"max_seq_len": int(max_len), "ppl": float(v)})
            logger.log(
                {
                    "run_timestamp": ts,
                    "model": model_id,
                    "model_slug": model_slug,
                    "branch": "quant",
                    "metric_family": "ppl_sweep",
                    "x_name": "max_seq_len",
                    "x_value": int(max_len),
                    "ppl": float(v),
                }
            )
            done_steps += 1
            progress(f"ppl_quant max_len={max_len}")
        ppl_q = ppl_sweep_quant[-1]["ppl"]

    q_runs: dict[int, dict] = {}
    for L in args.prefill_lens:
        print(f"[info] [quant] benchmarking prefill_len={L}")
        try:
            q_runs[L] = run_one_length(model, quantized=True, prefill_len=L, args=args, vocab_size=vocab_size)
        except Exception as e:
            print(f"[error] [quant] prefill_len={L} failed: {e}")
            q_runs[L] = {"error": str(e)}
            done_steps += 1
            progress(f"decode_quant prefill_len={L} error", force=True)
        else:
            done_steps += 1
            progress(f"decode_quant prefill_len={L}")

    cross: list[tuple[int, float, float]] = []
    cross_kv: list[tuple[int, float, float]] = []
    cross_vram: list[tuple[int, float, float]] = []

    for L in args.prefill_lens:
        ref = ref_runs.get(L)
        qnt = q_runs.get(L)
        per_len_dir = model_dir / f"prefill_{L}"
        per_len_dir.mkdir(parents=True, exist_ok=True)
        if ref is None or qnt is None or "error" in ref or "error" in qnt:
            err = {
                "prefill_len": L,
                "ref_error": ref.get("error") if ref else "missing",
                "quant_error": qnt.get("error") if qnt else "missing",
            }
            with open(per_len_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump(err, f, indent=2)
            continue

        metrics = _combine_length_metrics(L, args.num_decode_steps, ref, qnt)
        metrics.update({
            "run_timestamp": ts,
            "model": model_id,
            "cuda_description": cuda_desc,
            "group_size": args.group_size,
            "ppl_reference": ppl_ref,
            "ppl_quant": ppl_q,
        })
        _write_length_artifacts(per_len_dir, metrics)
        logger.log(
            {
                "run_timestamp": ts,
                "model": model_id,
                "model_slug": model_slug,
                "metric_family": "prefill_decode",
                "x_name": "prefill_len",
                "x_value": int(L),
                **metrics,
            }
        )
        cross.append((L, metrics["ref_decode_ms_per_token"], metrics["quant_decode_ms_per_token"]))
        cross_kv.append((L, metrics["ref_kv_cache_mib"], metrics["quant_kv_cache_mib"]))
        cross_vram.append((L, metrics["ref_peak_cuda_mib"], metrics["quant_peak_cuda_mib"]))
        print(
            f"[info] L={L}: ref {metrics['ref_decode_ms_per_token']:.2f} ms/tok, "
            f"quant {metrics['quant_decode_ms_per_token']:.2f} ms/tok, "
            f"KV ref {metrics['ref_kv_cache_mib']:.1f} MiB, "
            f"quant {metrics['quant_kv_cache_mib']:.1f} MiB"
        )

    passkey_summary: dict | None = None
    passkey_sweep: list[dict[str, Any]] | None = None
    longbench_summary: dict | None = None
    longbench_sweep: list[dict[str, Any]] | None = None

    # Two-phase retrieval: run quant branch now and merge with stored baseline results.
    if args.passkey_samples > 0 and passkey_sweep_tokens and passkey_ref_by_x:
        print(f"[info] RULER NIAH passkey sweep (quant) ctx_tokens={passkey_sweep_tokens}")
        try:
            pts: list[dict[str, Any]] = []
            for ctx_tokens in passkey_sweep_tokens:
                ref_blob = passkey_ref_by_x.get(int(ctx_tokens))
                if not ref_blob:
                    continue
                samples = ref_blob["samples"]
                ref_res = {r.sample_id: r for r in ref_blob["results"]}
                quant_res_list = run_passkey_branch(
                    model,
                    tokenizer,
                    _input_device(model),
                    samples,
                    cache_kind="quant",
                    group_size=args.group_size,
                    max_new_tokens=16,
                    quant_cache_cls=QuantizedKVCache,
                )
                hits_ref = 0
                hits_quant = 0
                parity = 0
                total_ctx = 0
                total_kv = 0.0
                n = max(1, len(samples))
                for qr in quant_res_list:
                    rr = ref_res.get(qr.sample_id)
                    if rr and rr.hit:
                        hits_ref += 1
                    if qr.hit:
                        hits_quant += 1
                    if rr and rr.answer_text.strip() == qr.answer_text.strip():
                        parity += 1
                    total_ctx += qr.n_in
                    if qr.kv_mib is not None:
                        total_kv += float(qr.kv_mib)
                pt = {
                    "haystack_tokens": int(ctx_tokens),
                    "seed": int(args.seed + 7),
                    "group_size": int(args.group_size),
                    "passkey_len": int(args.passkey_len),
                    "max_new_tokens": 16,
                    "n_samples": int(len(samples)),
                    "exact_match_rate_ref": hits_ref / n,
                    "exact_match_rate_quant": hits_quant / n,
                    "greedy_parity_rate": parity / n,
                    "avg_context_tokens": total_ctx / n,
                    "avg_quant_kv_mib": total_kv / n,
                    "dataset": "ruler_niah",
                }
                pts.append(pt)
                done_steps += 1
                progress(f"passkey_quant ctx_tokens={ctx_tokens}")
            passkey_sweep = pts
            with open(model_dir / "passkey_sweep.json", "w", encoding="utf-8") as f:
                json.dump({"points": passkey_sweep}, f, indent=2, default=str)
            for p in passkey_sweep:
                logger.log(
                    {
                        "run_timestamp": ts,
                        "model": model_id,
                        "model_slug": model_slug,
                        "metric_family": "passkey_sweep",
                        "x_name": "haystack_tokens",
                        "x_value": int(p["haystack_tokens"]),
                        **p,
                    }
                )
        except Exception as e:
            print(f"[error] passkey two-phase sweep failed: {e}")
            done_steps += 1
            progress("passkey_quant error", force=True)

    if longbench_subsets and args.longbench_max_samples > 0 and longbench_ctx_sweep and longbench_ref_by_x:
        print(f"[info] longbench sweep (quant) ctx_tokens={longbench_ctx_sweep} subsets={longbench_subsets}")
        try:
            pts: list[dict[str, Any]] = []
            for max_ctx in longbench_ctx_sweep:
                ref_blob = longbench_ref_by_x.get(int(max_ctx))
                if not ref_blob:
                    continue
                examples_by_subset = ref_blob["examples"]
                ref_by_subset = ref_blob["results"]
                by_subset_out: dict[str, dict[str, Any]] = {}
                all_ref_rates = []
                all_quant_rates = []
                all_parity_rates = []
                all_ns = []
                all_ctx = []
                all_kv = []
                for subset in longbench_subsets:
                    exs = examples_by_subset.get(subset, [])
                    ref_res_list = ref_by_subset.get(subset, [])
                    ref_map = {(r.subset, r.row_idx): r for r in ref_res_list}
                    quant_res_list = run_longbench_branch(
                        model,
                        tokenizer,
                        _input_device(model),
                        exs,
                        cache_kind="quant",
                        group_size=args.group_size,
                        max_new_tokens=int(args.longbench_max_new_tokens),
                        quant_cache_cls=QuantizedKVCache,
                    )
                    hits_ref = 0
                    hits_quant = 0
                    parity = 0
                    total_ctx = 0
                    total_kv = 0.0
                    n = max(1, len(exs))
                    for qr in quant_res_list:
                        rr = ref_map.get((qr.subset, qr.row_idx))
                        if rr and rr.hit:
                            hits_ref += 1
                        if qr.hit:
                            hits_quant += 1
                        if rr and rr.answer_text.strip() == qr.answer_text.strip():
                            parity += 1
                        total_ctx += qr.n_in
                        if qr.kv_mib is not None:
                            total_kv += float(qr.kv_mib)
                    by_subset_out[subset] = {
                        "subset": subset,
                        "n_samples": int(len(exs)),
                        "hit_rate_ref": hits_ref / n,
                        "hit_rate_quant": hits_quant / n,
                        "greedy_parity_rate": parity / n,
                        "avg_context_tokens": total_ctx / n,
                        "avg_quant_kv_mib": total_kv / n,
                    }
                    all_ref_rates.append(by_subset_out[subset]["hit_rate_ref"])
                    all_quant_rates.append(by_subset_out[subset]["hit_rate_quant"])
                    all_parity_rates.append(by_subset_out[subset]["greedy_parity_rate"])
                    all_ns.append(int(len(exs)))
                    all_ctx.append(by_subset_out[subset]["avg_context_tokens"])
                    all_kv.append(by_subset_out[subset]["avg_quant_kv_mib"])
                    done_steps += 1
                    progress(f"longbench_quant ctx={max_ctx} subset={subset}")

                mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
                total_n = sum(all_ns) or 1
                weighted = lambda xs: float(sum(float(x) * int(n) for x, n in zip(xs, all_ns)) / total_n) if xs else 0.0
                pt = {
                    "max_context_tokens": int(max_ctx),
                    "max_new_tokens": int(args.longbench_max_new_tokens),
                    "group_size": int(args.group_size),
                    "max_samples_per_subset": int(args.longbench_max_samples),
                    "subsets": list(longbench_subsets),
                    "hit_rate_ref_mean_subset": mean(all_ref_rates),
                    "hit_rate_quant_mean_subset": mean(all_quant_rates),
                    "greedy_parity_rate_mean_subset": mean(all_parity_rates),
                    "hit_rate_ref_mean_sample": weighted(all_ref_rates),
                    "hit_rate_quant_mean_sample": weighted(all_quant_rates),
                    "greedy_parity_rate_mean_sample": weighted(all_parity_rates),
                    "avg_context_tokens_mean_sample": weighted(all_ctx),
                    "avg_quant_kv_mib_mean_sample": weighted(all_kv),
                    "by_subset": by_subset_out,
                }
                pts.append(pt)
            longbench_sweep = pts
            with open(model_dir / "longbench_sweep.json", "w", encoding="utf-8") as f:
                json.dump({"points": longbench_sweep}, f, indent=2, default=str)
            for p in longbench_sweep:
                logger.log(
                    {
                        "run_timestamp": ts,
                        "model": model_id,
                        "model_slug": model_slug,
                        "metric_family": "longbench_sweep",
                        "x_name": "max_context_tokens",
                        "x_value": int(p["max_context_tokens"]),
                        **p,
                    }
                )
        except Exception as e:
            print(f"[error] longbench two-phase sweep failed: {e}")
            done_steps += 1
            progress("longbench_quant error", force=True)

    if texts and ppl_sweep_lens and ppl_sweep_ref and ppl_sweep_quant:
        with open(model_dir / "ppl_sweep.json", "w", encoding="utf-8") as f:
            json.dump(
                {"points": [{"max_seq_len": r["max_seq_len"], "ref_ppl": r["ppl"], "quant_ppl": q["ppl"]}
                            for r, q in zip(ppl_sweep_ref, ppl_sweep_quant)]},
                f,
                indent=2,
                default=str,
            )
        _plot_cross_length_multi(
            model_dir / "ppl_by_max_seq_len.png",
            f"{model_slug}: perplexity vs max sequence length",
            "perplexity (lower is better)",
            [{"max_seq_len": r["max_seq_len"], "ref": r["ppl"], "quant": q["ppl"]} for r, q in zip(ppl_sweep_ref, ppl_sweep_quant)],
            x_name="max_seq_len",
            y_name="ppl",
            ref_key="ref",
            quant_key="quant",
        )

    if passkey_sweep:
        _plot_cross_length_multi(
            model_dir / "passkey_hit_rate_by_haystack_tokens.png",
            f"{model_slug}: passkey exact match vs haystack tokens",
            "exact match rate",
            [{"haystack_tokens": p["haystack_tokens"], "ref": p["exact_match_rate_ref"], "quant": p["exact_match_rate_quant"]} for p in passkey_sweep],
            x_name="haystack_tokens",
            y_name="exact_match_rate",
            ref_key="ref",
            quant_key="quant",
        )

    if longbench_sweep:
        _plot_cross_length_multi(
            model_dir / "longbench_hit_rate_mean_by_context_tokens.png",
            f"{model_slug}: long retrieval hit rate vs context tokens",
            "hit rate (mean across subsets)",
            [{"max_context_tokens": p["max_context_tokens"], "ref": p["hit_rate_ref_mean_subset"], "quant": p["hit_rate_quant_mean_subset"]} for p in longbench_sweep],
            x_name="max_context_tokens",
            y_name="hit_rate_mean",
            ref_key="ref",
            quant_key="quant",
        )

    _plot_cross_length(
        model_dir / "decode_ms_per_token_by_seqlen.png",
        f"{model_slug}: decode latency vs prefill length",
        "ms / token",
        cross,
    )
    _plot_cross_length(
        model_dir / "kv_cache_storage_mib_by_seqlen.png",
        f"{model_slug}: KV tensor storage vs prefill length",
        "MiB",
        cross_kv,
    )
    _plot_cross_length(
        model_dir / "peak_vram_mib_by_seqlen.png",
        f"{model_slug}: peak CUDA allocated (sum across GPUs) vs prefill length",
        "MiB",
        cross_vram,
    )

    model_summary.update({
        "ppl_reference": ppl_ref,
        "ppl_quant": ppl_q,
        "passkey": passkey_summary,
        "longbench": longbench_summary,
        "ppl_sweep": (
            [{"max_seq_len": r["max_seq_len"], "ref_ppl": r["ppl"], "quant_ppl": q["ppl"]} for r, q in zip(ppl_sweep_ref, ppl_sweep_quant)]
            if (ppl_sweep_ref and ppl_sweep_quant)
            else None
        ),
        "passkey_sweep": passkey_sweep,
        "longbench_sweep": longbench_sweep,
        "cross_decode_ms_per_token": [
            {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in cross
        ],
        "cross_kv_cache_mib": [
            {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in cross_kv
        ],
        "cross_peak_vram_mib": [
            {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in cross_vram
        ],
    })
    with open(model_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2, default=str)
    logger.close()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_id,
        "model_slug": model_slug,
        "cross_decode_ms_per_token": cross,
        "cross_kv_cache_mib": cross_kv,
        "cross_peak_vram_mib": cross_vram,
        "ppl_reference": ppl_ref,
        "ppl_quant": ppl_q,
        "passkey": passkey_summary,
        "longbench": longbench_summary,
        "ppl_sweep": [
            (int(r["max_seq_len"]), float(r["ppl"]), float(q["ppl"]))
            for r, q in zip(ppl_sweep_ref, ppl_sweep_quant)
        ]
        if (ppl_sweep_ref and ppl_sweep_quant)
        else None,
        "passkey_sweep": [
            (int(p["haystack_tokens"]), float(p["exact_match_rate_ref"]), float(p["exact_match_rate_quant"]))
            for p in (passkey_sweep or [])
        ]
        if passkey_sweep
        else None,
        "longbench_sweep": [
            (int(p["max_context_tokens"]), float(p["hit_rate_ref_mean_subset"]), float(p["hit_rate_quant_mean_subset"]))
            for p in (longbench_sweep or [])
        ]
        if longbench_sweep
        else None,
    }


def _describe_cuda() -> str:
    n = _num_devices()
    if n == 0:
        return "cpu"
    names = {torch.cuda.get_device_name(d) for d in range(n)}
    if len(names) == 1:
        return f"{n}x {next(iter(names))}"
    return ", ".join(f"gpu{d}={torch.cuda.get_device_name(d)}" for d in range(n))


def main() -> None:
    _ensure_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"HF model ids to benchmark in order (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--prefill-lens",
        type=int,
        nargs="+",
        default=DEFAULT_PREFILL_LENS,
        help="Prefill lengths to sweep (tokens).",
    )
    parser.add_argument("--num-decode-steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppl-samples", type=int, default=32)
    parser.add_argument("--ppl-max-seq-len", type=int, default=512)
    parser.add_argument(
        "--ppl-max-seq-len-sweep",
        type=int,
        nargs="*",
        default=[],
        help="Optional sweep of PPL max_seq_len values (e.g. 128 256 512 1024). If provided, PPL is computed for each value.",
    )
    parser.add_argument("--passkey-samples", type=int, default=0)
    parser.add_argument("--passkey-haystack-tokens", type=int, default=4096)
    parser.add_argument(
        "--passkey-haystack-sweep",
        type=int,
        nargs="*",
        default=[],
        help="Optional sweep of passkey haystack tokens (e.g. 1024 2048 4096 8192).",
    )
    parser.add_argument("--passkey-len", type=int, default=8)
    parser.add_argument(
        "--longbench-subsets",
        type=str,
        default="",
        help="Comma-separated LongBench subsets (e.g. narrativeqa,triviaqa). Empty disables.",
    )
    parser.add_argument("--longbench-max-samples", type=int, default=0)
    parser.add_argument("--longbench-max-context-tokens", type=int, default=8192)
    parser.add_argument(
        "--longbench-context-sweep",
        type=int,
        nargs="*",
        default=[],
        help="Optional sweep of LongBench max_context_tokens (e.g. 2048 4096 8192 16384).",
    )
    parser.add_argument("--longbench-max-new-tokens", type=int, default=48)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a model fails to load or bench, record the error and continue to the next.",
    )
    parser.add_argument(
        "--kernel-variant",
        choices=("auto", "default", "sm70"),
        default="auto",
        help=(
            "Which Triton kernels to use. 'default' uses kernels/llama3/{attention_quant,quantize}.py "
            "(requires sm_80+ because Triton emits .bf16 PTX). 'sm70' uses the Volta-safe copies in "
            "kernels/llama3/*_sm70.py (casts bf16 to fp16 at the Triton boundary). 'auto' picks sm70 "
            "when any visible GPU is below sm_80."
        ),
    )
    args = parser.parse_args()

    kernel_variant = _resolve_kernel_variant(args.kernel_variant)
    _bind_kernel_variant(kernel_variant)
    print(
        f"[info] kernel variant: {kernel_variant}"
        + (" (FP16 boundary, Volta-safe)" if kernel_variant == "sm70" else "")
    )
    print(f"[info] bound QuantizedKVCache: {QuantizedKVCache.__module__}.{QuantizedKVCache.__name__}")
    # Unmissable routing banner. Silent misconfiguration (benchmark running with
    # an unintended fused path) caused a multi-hour debug loop; always log what
    # the quant attention module captured at import time so the active routing
    # decision is visible in every benchmark log.
    import os as _os
    _fallback = _os.environ.get("KV_FORCE_DEQUANT_FALLBACK", "") == "1"
    _branch = "dequant_SDPA (fused kernel bypassed)" if _fallback else f"fused ({kernel_variant})"
    print(f"[info] KV_FORCE_DEQUANT_FALLBACK={'1' if _fallback else '0'} -> attention branch: {_branch}")
    if kernel_variant == "sm70" and not _fallback:
        print(
            "[info] running fused sm_70 attention kernel; set KV_FORCE_DEQUANT_FALLBACK=1 "
            "before Python to force the dequant+SDPA reference path instead."
        )
    if _os.environ.get("KV_PASSTHROUGH_BF16", "") == "1":
        print("[info] KV_PASSTHROUGH_BF16=1 -> cache stores raw bf16 K/V, quantize/dequantize skipped")

    model_ids: list[str] = args.models if args.models is not None else list(DEFAULT_MODELS)
    longbench_subsets = [s.strip() for s in args.longbench_subsets.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (_ROOT / "benchmarks" / "llama3" / "logs" / ts)
    out_root.mkdir(parents=True, exist_ok=True)

    cuda_desc = _describe_cuda()
    root_logger = BenchLogger(out_root)
    print(f"[info] CUDA: {cuda_desc}")
    print(f"[info] writing to {out_root}")
    print(f"[info] models: {model_ids}")
    print(f"[info] prefill lens: {args.prefill_lens}")

    root_logger.write_config(
        {
            "timestamp": ts,
            "cuda_description": cuda_desc,
            "prefill_lens": list(args.prefill_lens),
            "group_size": args.group_size,
            "num_decode_steps": args.num_decode_steps,
            "seed": args.seed,
            "ppl_samples": args.ppl_samples,
            "ppl_max_seq_len": args.ppl_max_seq_len,
            "passkey_samples": args.passkey_samples,
            "passkey_haystack_tokens": args.passkey_haystack_tokens,
            "passkey_len": args.passkey_len,
            "longbench_subsets": longbench_subsets,
            "longbench_max_samples": args.longbench_max_samples,
            "longbench_max_context_tokens": args.longbench_max_context_tokens,
            "longbench_max_new_tokens": args.longbench_max_new_tokens,
            "kernel_variant": kernel_variant,
            "kv_force_dequant_fallback": _fallback,
            "attention_branch": _branch,
        }
    )

    per_model_decode: dict[str, list[tuple[int, float, float]]] = {}
    per_model_kv: dict[str, list[tuple[int, float, float]]] = {}
    per_model_vram: dict[str, list[tuple[int, float, float]]] = {}
    per_model_ppl: dict[str, list[tuple[int, float, float]]] = {}
    per_model_passkey: dict[str, list[tuple[int, float, float]]] = {}
    per_model_longbench: dict[str, list[tuple[int, float, float]]] = {}
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for model_id in model_ids:
        try:
            res = run_one_model(
                model_id=model_id,
                args=args,
                out_root=out_root,
                ts=ts,
                cuda_desc=cuda_desc,
                longbench_subsets=longbench_subsets,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[error] model {model_id} failed: {msg}")
            model_dir = out_root / _model_slug(model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump({"model": model_id, "error": msg}, f, indent=2)
            failed.append({"model": model_id, "error": msg})
            gc.collect()
            torch.cuda.empty_cache()
            if args.continue_on_error:
                continue
            raise
        else:
            completed.append(res)
            per_model_decode[res["model_slug"]] = res["cross_decode_ms_per_token"]
            per_model_kv[res["model_slug"]] = res["cross_kv_cache_mib"]
            per_model_vram[res["model_slug"]] = res["cross_peak_vram_mib"]
            if res.get("ppl_sweep"):
                per_model_ppl[res["model_slug"]] = res["ppl_sweep"]
            if res.get("passkey_sweep"):
                per_model_passkey[res["model_slug"]] = res["passkey_sweep"]
            if res.get("longbench_sweep"):
                per_model_longbench[res["model_slug"]] = res["longbench_sweep"]

    _plot_cross_models(
        out_root / "decode_ms_per_token_by_size.png",
        "Decode latency by model (BF16 vs INT4 KV)",
        "ms / token",
        per_model_decode,
    )
    _plot_cross_models(
        out_root / "kv_cache_storage_mib_by_size.png",
        "KV cache tensor storage by model",
        "MiB",
        per_model_kv,
    )
    _plot_cross_models(
        out_root / "peak_vram_mib_by_size.png",
        "Peak CUDA allocated (sum across GPUs) by model",
        "MiB",
        per_model_vram,
    )
    _plot_cross_models(
        out_root / "ppl_by_model.png",
        "Perplexity vs max sequence length (BF16 vs INT4 KV)",
        "perplexity",
        per_model_ppl,
    )
    _plot_cross_models(
        out_root / "passkey_hit_rate_by_model.png",
        "Passkey exact match vs haystack tokens (BF16 vs INT4 KV)",
        "exact match rate",
        per_model_passkey,
    )
    _plot_cross_models(
        out_root / "longbench_hit_rate_by_model.png",
        "Long retrieval hit rate vs context tokens (BF16 vs INT4 KV)",
        "hit rate (mean across subsets)",
        per_model_longbench,
    )

    summary = {
        "timestamp": ts,
        "cuda_description": cuda_desc,
        "prefill_lens": args.prefill_lens,
        "group_size": args.group_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_models": model_ids,
        "completed": [
            {
                "model": r["model"],
                "ppl_reference": r["ppl_reference"],
                "ppl_quant": r["ppl_quant"],
                "cross_decode_ms_per_token": [
                    {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in r["cross_decode_ms_per_token"]
                ],
                "cross_kv_cache_mib": [
                    {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in r["cross_kv_cache_mib"]
                ],
                "cross_peak_vram_mib": [
                    {"prefill_len": L, "ref": a, "quant": b} for (L, a, b) in r["cross_peak_vram_mib"]
                ],
            }
            for r in completed
        ],
        "failed": failed,
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[info] wrote {out_root}")
    if failed:
        print(f"[warn] {len(failed)} model(s) failed: {[f['model'] for f in failed]}")
    root_logger.log({"run_timestamp": ts, "metric_family": "run_summary", **summary})
    root_logger.close()


if __name__ == "__main__":
    main()
