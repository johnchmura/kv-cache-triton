"""Benchmark Llama 3.1 8B baseline (BF16 KV) vs quantized (INT4 KV).

Runs a configurable prefill/decode sweep with per-length metrics (decode
latency, peak VRAM, KV tensor bytes, tokens/s) plus optional quality evals
(WikiText-2 PPL, passkey retrieval, LongBench). Writes artifacts under
``benchmarks/llama3/logs/<ts>/``.

Baseline and quantized paths are intentionally run in series inside the same
process to keep load latency amortized. Callers sweeping many prefill lengths
should launch separate processes (e.g. one shell subagent per length) to avoid
HBM fragmentation between runs.
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
from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.llama3_quant import replace_llama_attention_with_quantized

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_PREFILL_LENS = [1024, 4096, 16384]
DEFAULT_LONGBENCH_SUBSETS = ["narrativeqa", "triviaqa"]


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")


def _load_llama(model_id: str, device: torch.device, quantized: bool, group_size: int):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    if quantized:
        replace_llama_attention_with_quantized(model, group_size=group_size)
    return model


def _make_cache(quantized: bool, group_size: int):
    return QuantizedKVCache(group_size=group_size) if quantized else DynamicCache()


def _benchmark_decode(
    model,
    device: torch.device,
    prefill_len: int,
    num_decode_steps: int,
    warmup: int,
    seed: int,
    quantized: bool,
    group_size: int,
    vocab_size: int,
) -> dict:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, prefill_len), device=device, dtype=torch.long)

    def _run_once(timed: bool) -> dict:
        if timed:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = _make_cache(quantized, group_size)
        with torch.inference_mode():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        t2 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_decode_steps):
                out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
                next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        res = {
            "prefill_ms": (t1 - t0) * 1000.0,
            "decode_ms": (t3 - t2) * 1000.0,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
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
    device: torch.device,
    max_length: int,
    max_batches: int,
    quantized: bool,
    group_size: int,
) -> float:
    model.eval()
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
        input_ids = enc["input_ids"].to(device)
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


def _write_artifacts(out_dir: Path, metrics: dict[str, Any]) -> None:
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
        "Peak CUDA memory during decode",
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
    if metrics.get("ppl_reference") is not None and metrics.get("ppl_quant") is not None:
        _plot_bars(
            out_dir / "perplexity.png",
            "Perplexity on WikiText-2 validation (subset)",
            "PPL",
            ["reference", "quantized"],
            [metrics["ppl_reference"], metrics["ppl_quant"]],
        )
    if metrics.get("passkey_n_samples"):
        _plot_bars(
            out_dir / "passkey_match.png",
            "Passkey retrieval (substring match)",
            "rate",
            ["reference", "quantized"],
            [metrics["passkey_exact_match_rate_ref"], metrics["passkey_exact_match_rate_quant"]],
        )
    if metrics.get("longbench_hit_rate_ref_mean") is not None:
        _plot_bars(
            out_dir / "longbench_hit.png",
            "LongBench (tail-truncated) substring hit",
            "rate",
            ["reference", "quantized"],
            [metrics["longbench_hit_rate_ref_mean"], metrics["longbench_hit_rate_quant_mean"]],
        )


def run_one_length(
    model_ref,
    model_q,
    prefill_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    vocab_size = model_ref.config.vocab_size

    ref = _benchmark_decode(
        model_ref, device, prefill_len, args.num_decode_steps, args.warmup,
        seed=args.seed, quantized=False, group_size=args.group_size, vocab_size=vocab_size,
    )
    qnt = _benchmark_decode(
        model_q, device, prefill_len, args.num_decode_steps, args.warmup,
        seed=args.seed + 3, quantized=True, group_size=args.group_size, vocab_size=vocab_size,
    )
    return {
        "prefill_len": prefill_len,
        "num_decode_steps": args.num_decode_steps,
        "ref_prefill_ms": ref["prefill_ms"],
        "ref_decode_total_ms": ref["decode_ms"],
        "ref_decode_ms_per_token": ref["decode_ms"] / args.num_decode_steps,
        "ref_tokens_per_s": args.num_decode_steps / (ref["decode_ms"] / 1000.0),
        "ref_peak_cuda_mib": ref["peak_cuda_mib"],
        "ref_kv_cache_bytes": ref["kv_bytes"],
        "ref_kv_cache_mib": ref["kv_bytes"] / 1024**2,
        "quant_prefill_ms": qnt["prefill_ms"],
        "quant_decode_total_ms": qnt["decode_ms"],
        "quant_decode_ms_per_token": qnt["decode_ms"] / args.num_decode_steps,
        "quant_tokens_per_s": args.num_decode_steps / (qnt["decode_ms"] / 1000.0),
        "quant_peak_cuda_mib": qnt["peak_cuda_mib"],
        "quant_kv_cache_bytes": qnt["kv_bytes"],
        "quant_kv_cache_mib": qnt["kv_bytes"] / 1024**2,
    }


def main() -> None:
    _ensure_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    args = parser.parse_args()

    longbench_subsets = [s.strip() for s in args.longbench_subsets.split(",") if s.strip()]

    device = torch.device("cuda")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (_ROOT / "benchmarks" / "llama3" / "logs" / ts)
    out_root.mkdir(parents=True, exist_ok=True)

    cuda_name = torch.cuda.get_device_name(0)
    print(f"[info] CUDA: {cuda_name}")
    print(f"[info] writing to {out_root}")
    print(f"[info] sweeping prefill lens: {args.prefill_lens}")

    print(f"[info] loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[info] loading baseline (BF16 KV)")
    model_ref = _load_llama(args.model, device, quantized=False, group_size=args.group_size)

    texts = _load_wikitext(args.ppl_samples) if args.ppl_samples > 0 else []

    ppl_ref = None
    if texts:
        print(f"[info] computing reference PPL on {len(texts)} WikiText-2 rows")
        ppl_ref = _compute_ppl(
            model_ref, tokenizer, texts, device, args.ppl_max_seq_len,
            max_batches=args.ppl_samples, quantized=False, group_size=args.group_size,
        )
        print(f"[info] ref PPL: {ppl_ref:.3f}")

    print(f"[info] loading quantized (INT4 KV, group_size={args.group_size})")
    model_q = _load_llama(args.model, device, quantized=True, group_size=args.group_size)

    ppl_q = None
    if texts:
        print(f"[info] computing quantized PPL")
        ppl_q = _compute_ppl(
            model_q, tokenizer, texts, device, args.ppl_max_seq_len,
            max_batches=args.ppl_samples, quantized=True, group_size=args.group_size,
        )
        print(f"[info] quant PPL: {ppl_q:.3f}")

    cross: list[tuple[int, float, float]] = []
    cross_kv: list[tuple[int, float, float]] = []
    cross_vram: list[tuple[int, float, float]] = []

    for L in args.prefill_lens:
        print(f"[info] benchmarking prefill_len={L}")
        per_len_dir = out_root / f"prefill_{L}"
        per_len_dir.mkdir(parents=True, exist_ok=True)
        try:
            metrics = run_one_length(model_ref, model_q, L, args, device)
        except Exception as e:
            print(f"[error] prefill_len={L} failed: {e}")
            with open(per_len_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump({"prefill_len": L, "error": str(e)}, f, indent=2)
            continue
        metrics.update(
            {
                "run_timestamp": ts,
                "model": args.model,
                "cuda_device_name": cuda_name,
                "group_size": args.group_size,
                "ppl_reference": ppl_ref,
                "ppl_quant": ppl_q,
            }
        )
        _write_artifacts(per_len_dir, metrics)
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
    if args.passkey_samples > 0:
        print(f"[info] passkey eval x {args.passkey_samples}")
        pk = run_passkey_eval(
            model_ref, model_q, tokenizer, device,
            group_size=args.group_size,
            n_samples=args.passkey_samples,
            passkey_len=args.passkey_len,
            haystack_tokens=args.passkey_haystack_tokens,
            seed=args.seed + 7,
        )
        passkey_summary = asdict(pk)
        with open(out_root / "passkey.json", "w", encoding="utf-8") as f:
            json.dump(passkey_summary, f, indent=2)

    longbench_summary: dict | None = None
    if longbench_subsets and args.longbench_max_samples > 0:
        print(f"[info] longbench eval subsets={longbench_subsets}")
        lb = run_longbench_eval(
            model_ref, model_q, tokenizer, device,
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
        with open(out_root / "longbench.json", "w", encoding="utf-8") as f:
            json.dump(longbench_summary, f, indent=2)

    _plot_cross_length(
        out_root / "decode_ms_per_token_by_seqlen.png",
        "Decode latency vs prefill length",
        "ms / token",
        cross,
    )
    _plot_cross_length(
        out_root / "kv_cache_storage_mib_by_seqlen.png",
        "KV tensor storage vs prefill length",
        "MiB",
        cross_kv,
    )
    _plot_cross_length(
        out_root / "peak_vram_mib_by_seqlen.png",
        "Peak CUDA allocated vs prefill length",
        "MiB",
        cross_vram,
    )

    summary = {
        "timestamp": ts,
        "model": args.model,
        "cuda_device_name": cuda_name,
        "prefill_lens": args.prefill_lens,
        "group_size": args.group_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
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
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[info] wrote {out_root}")


if __name__ == "__main__":
    main()
