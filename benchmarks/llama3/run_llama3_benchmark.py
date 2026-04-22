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

from benchmarks.llama3.eval_longbench import run_longbench_eval
from benchmarks.llama3.eval_passkey import run_passkey_eval
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

    ref_runs: dict[int, dict] = {}

    print("[info] --- reference pass (stock DynamicCache, BF16 KV) ---")
    ppl_ref = None
    if texts:
        print(f"[info] computing reference PPL on {len(texts)} WikiText-2 rows")
        ppl_ref = _compute_ppl(
            model, tokenizer, texts, args.ppl_max_seq_len,
            max_batches=args.ppl_samples, quantized=False, group_size=args.group_size,
        )
        print(f"[info] ref PPL: {ppl_ref:.3f}")

    for L in args.prefill_lens:
        print(f"[info] [ref]  benchmarking prefill_len={L}")
        try:
            ref_runs[L] = run_one_length(model, quantized=False, prefill_len=L, args=args, vocab_size=vocab_size)
        except Exception as e:
            print(f"[error] [ref]  prefill_len={L} failed: {e}")
            ref_runs[L] = {"error": str(e)}

    print("[info] --- swapping attention modules to quantized INT4 path ---")
    replace_llama_attention_with_quantized(model, group_size=args.group_size)

    ppl_q = None
    if texts:
        print(f"[info] computing quantized PPL")
        ppl_q = _compute_ppl(
            model, tokenizer, texts, args.ppl_max_seq_len,
            max_batches=args.ppl_samples, quantized=True, group_size=args.group_size,
        )
        print(f"[info] quant PPL: {ppl_q:.3f}")

    q_runs: dict[int, dict] = {}
    for L in args.prefill_lens:
        print(f"[info] [quant] benchmarking prefill_len={L}")
        try:
            q_runs[L] = run_one_length(model, quantized=True, prefill_len=L, args=args, vocab_size=vocab_size)
        except Exception as e:
            print(f"[error] [quant] prefill_len={L} failed: {e}")
            q_runs[L] = {"error": str(e)}

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
    longbench_summary: dict | None = None

    if args.passkey_samples > 0:
        print(f"[info] passkey eval x {args.passkey_samples}  (note: shares one swapped model for both paths)")
        try:
            pk = run_passkey_eval(
                model, model, tokenizer, _input_device(model),
                group_size=args.group_size,
                n_samples=args.passkey_samples,
                passkey_len=args.passkey_len,
                haystack_tokens=args.passkey_haystack_tokens,
                seed=args.seed + 7,
            )
            passkey_summary = asdict(pk)
            with open(model_dir / "passkey.json", "w", encoding="utf-8") as f:
                json.dump(passkey_summary, f, indent=2)
        except Exception as e:
            print(f"[error] passkey eval failed: {e}")

    if longbench_subsets and args.longbench_max_samples > 0:
        print(f"[info] longbench eval subsets={longbench_subsets}")
        try:
            lb = run_longbench_eval(
                model, model, tokenizer, _input_device(model),
                group_size=args.group_size,
                subsets=longbench_subsets,
                max_samples_per_subset=args.longbench_max_samples,
                max_context_tokens=args.longbench_max_context_tokens,
                max_new_tokens=args.longbench_max_new_tokens,
            )
            longbench_summary = {
                "hit_rate_ref_mean": lb.hit_rate_ref_mean,
                "hit_rate_quant_mean": lb.hit_rate_quant_mean,
                "greedy_parity_rate_mean": lb.greedy_parity_rate_mean,
                "results": [asdict(r) for r in lb.results],
                "max_context_tokens": lb.max_context_tokens,
                "max_new_tokens": lb.max_new_tokens,
            }
            with open(model_dir / "longbench.json", "w", encoding="utf-8") as f:
                json.dump(longbench_summary, f, indent=2)
        except Exception as e:
            print(f"[error] longbench eval failed: {e}")

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
    parser.add_argument("--passkey-samples", type=int, default=0)
    parser.add_argument("--passkey-haystack-tokens", type=int, default=4096)
    parser.add_argument("--passkey-len", type=int, default=8)
    parser.add_argument(
        "--longbench-subsets",
        type=str,
        default="",
        help="Comma-separated LongBench subsets (e.g. narrativeqa,triviaqa). Empty disables.",
    )
    parser.add_argument("--longbench-max-samples", type=int, default=0)
    parser.add_argument("--longbench-max-context-tokens", type=int, default=8192)
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
    print(f"[info] CUDA: {cuda_desc}")
    print(f"[info] writing to {out_root}")
    print(f"[info] models: {model_ids}")
    print(f"[info] prefill lens: {args.prefill_lens}")

    per_model_decode: dict[str, list[tuple[int, float, float]]] = {}
    per_model_kv: dict[str, list[tuple[int, float, float]]] = {}
    per_model_vram: dict[str, list[tuple[int, float, float]]] = {}
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


if __name__ == "__main__":
    main()
