"""Benchmark GPT-2 with Hugging Face attention vs Triton decode attention."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.gpt2_triton import replace_gpt2_attention_with_triton

DEFAULT_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")


def _run_prefill_decode(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    num_decode_steps: int,
) -> None:
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_id = torch.randint(0, 50257, (1, 1), device=input_ids.device, dtype=torch.long)
        for _ in range(num_decode_steps):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = torch.randint(0, 50257, (1, 1), device=input_ids.device, dtype=torch.long)


def benchmark_decode(
    model: GPT2LMHeadModel,
    device: torch.device,
    prefill_len: int,
    num_decode_steps: int,
    warmup: int,
    seed: int,
) -> tuple[float, float, float]:
    """Returns prefill_ms, decode_total_ms, peak_alloc_mib."""
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 50257, (1, prefill_len), device=device, dtype=torch.long)

    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _run_prefill_decode(model, input_ids, num_decode_steps)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        past = out.past_key_values
        next_id = torch.randint(0, 50257, (1, 1), device=device, dtype=torch.long)
        t2 = time.perf_counter()
        for _ in range(num_decode_steps):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = torch.randint(0, 50257, (1, 1), device=device, dtype=torch.long)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

    prefill_ms = (t1 - t0) * 1000.0
    decode_ms = (t3 - t2) * 1000.0
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return prefill_ms, decode_ms, peak_mib


def compute_ppl(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    max_batches: int,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    for i in range(0, len(texts), batch_size):
        if n_batches >= max_batches:
            break
        batch_texts = texts[i : i + batch_size]
        if not batch_texts:
            continue
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        num_valid = (labels != -100).sum().item()
        total_nll += loss.item() * num_valid
        total_tokens += num_valid
        n_batches += 1
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


def logit_parity_metrics(
    model_a: GPT2LMHeadModel,
    model_b: GPT2LMHeadModel,
    device: torch.device,
    prefill_len: int,
    decode_steps: int,
    seed: int,
) -> tuple[float, float]:
    """Returns max_abs_logits_diff, mean_abs_logits_diff on last forward."""
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 50257, (1, prefill_len), device=device, dtype=torch.long)

    with torch.no_grad():
        out_a = model_a(input_ids=input_ids, use_cache=True)
        past_a = out_a.past_key_values
        out_b = model_b(input_ids=input_ids, use_cache=True)
        past_b = out_b.past_key_values
        next_id = torch.randint(0, 50257, (1, 1), device=device, dtype=torch.long)
        for _ in range(decode_steps):
            out_a = model_a(input_ids=next_id, past_key_values=past_a, use_cache=True)
            past_a = out_a.past_key_values
            out_b = model_b(input_ids=next_id, past_key_values=past_b, use_cache=True)
            past_b = out_b.past_key_values
            next_id = torch.randint(0, 50257, (1, 1), device=device, dtype=torch.long)
        logits_a = out_a.logits
        logits_b = out_b.logits
    diff = (logits_a - logits_b).abs()
    return diff.max().item(), diff.mean().item()


def load_gpt2_half(device: torch.device, model_id: str) -> GPT2LMHeadModel:
    m = GPT2LMHeadModel.from_pretrained(model_id, dtype=torch.float16)
    m = m.to(device)
    m.eval()
    return m


def plot_grouped_bars(
    path: Path,
    title: str,
    ylabel: str,
    left_label: str,
    right_label: str,
    left_val: float,
    right_val: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0, 1]
    ax.bar(x, [left_val, right_val], width=0.5, color=["#4477AA", "#DDAA33"])
    ax.set_xticks(x)
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_agreement_bars(path: Path, max_abs: float, rel_ppl_pct: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["max |logits ref - triton|", "rel. |PPL delta| (%)"]
    ax.bar([0, 1], [max_abs, rel_ppl_pct], width=0.5, color=["#AA4455", "#66C2A5"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Numerical agreement (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cross_model_decode_latency(
    path: Path,
    rows: list[tuple[str, float, float]],
) -> None:
    """rows: (model_id, ref_ms_per_token, triton_ms_per_token)."""
    if not rows:
        return
    labels = [r[0] for r in rows]
    ref_v = [r[1] for r in rows]
    tr_v = [r[2] for r in rows]
    n = len(labels)
    idx = list(range(n))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n), 4))
    ax.bar([i - width / 2 for i in idx], ref_v, width, label="reference", color="#4477AA")
    ax.bar([i + width / 2 for i in idx], tr_v, width, label="triton", color="#DDAA33")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ms per token")
    ax.set_title("Decode latency by model (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_artifacts(out_dir: Path, metrics: dict[str, Any]) -> None:
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, (int, float, str)):
                w.writerow([k, v])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ms_per_tok_ref = metrics["ref_decode_ms_per_token"]
    ms_per_tok_tr = metrics["triton_decode_ms_per_token"]
    ref_peak = metrics["ref_peak_cuda_mib"]
    tr_peak = metrics["triton_peak_cuda_mib"]
    ppl_ref = metrics["ppl_reference"]
    ppl_tr = metrics["ppl_triton"]
    max_logit_diff = metrics["max_abs_logits_diff"]
    rel_ppl_pct = metrics["rel_ppl_delta_pct"]

    plot_grouped_bars(
        out_dir / "decode_latency.png",
        "Decode latency (lower is better)",
        "ms per token",
        "reference",
        "triton",
        float(ms_per_tok_ref),
        float(ms_per_tok_tr),
    )
    plot_grouped_bars(
        out_dir / "peak_vram.png",
        "Peak CUDA memory (decode benchmark run)",
        "MiB (max allocated)",
        "reference",
        "triton",
        float(ref_peak),
        float(tr_peak),
    )
    plot_grouped_bars(
        out_dir / "throughput.png",
        "Decode throughput (higher is better)",
        "tokens/s",
        "reference",
        "triton",
        float(metrics["ref_tokens_per_s"]),
        float(metrics["triton_tokens_per_s"]),
    )
    plot_grouped_bars(
        out_dir / "perplexity.png",
        "Perplexity on WikiText-2 validation (subset)",
        "PPL",
        "reference",
        "triton",
        float(ppl_ref) if ppl_ref is not None else 0.0,
        float(ppl_tr) if ppl_tr is not None else 0.0,
    )
    plot_agreement_bars(out_dir / "logit_agreement.png", float(max_logit_diff), float(rel_ppl_pct))


def run_single_checkpoint(
    model_id: str,
    out_dir: Path,
    *,
    device: torch.device,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    run_timestamp: str,
    cuda_name: str,
    args: Namespace,
) -> dict[str, Any]:
    """Runs ref vs Triton benchmark for one HF checkpoint; writes CSV/JSON/PNGs to out_dir."""
    torch.manual_seed(args.seed)

    model_ref = load_gpt2_half(device, model_id)
    ref_prefill_ms, ref_decode_ms, ref_peak = benchmark_decode(
        model_ref,
        device,
        args.prefill_len,
        args.num_decode_steps,
        args.warmup,
        args.seed,
    )
    ppl_ref = compute_ppl(
        model_ref,
        tokenizer,
        texts,
        device,
        args.max_seq_len,
        args.batch_size,
        args.max_eval_batches,
    )
    del model_ref
    gc.collect()
    torch.cuda.empty_cache()

    model_tr = load_gpt2_half(device, model_id)
    replace_gpt2_attention_with_triton(model_tr)
    tr_prefill_ms, tr_decode_ms, tr_peak = benchmark_decode(
        model_tr,
        device,
        args.prefill_len,
        args.num_decode_steps,
        args.warmup,
        args.seed + 1,
    )
    ppl_tr = compute_ppl(
        model_tr,
        tokenizer,
        texts,
        device,
        args.max_seq_len,
        args.batch_size,
        args.max_eval_batches,
    )

    model_ref2 = load_gpt2_half(device, model_id)
    max_logit_diff, mean_logit_diff = logit_parity_metrics(
        model_ref2,
        model_tr,
        device,
        args.prefill_len,
        args.logit_decode_steps,
        args.seed + 2,
    )
    del model_ref2
    gc.collect()
    torch.cuda.empty_cache()

    del model_tr
    gc.collect()
    torch.cuda.empty_cache()

    ms_per_tok_ref = ref_decode_ms / args.num_decode_steps
    ms_per_tok_tr = tr_decode_ms / args.num_decode_steps
    tps_ref = args.num_decode_steps / (ref_decode_ms / 1000.0)
    tps_tr = args.num_decode_steps / (tr_decode_ms / 1000.0)

    rel_ppl_pct = 0.0
    if ppl_ref is not None and ppl_tr is not None and ppl_ref > 0:
        rel_ppl_pct = abs(ppl_ref - ppl_tr) / ppl_ref * 100.0

    metrics: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "model_id": model_id,
        "cuda_device_name": cuda_name,
        "prefill_len": args.prefill_len,
        "num_decode_steps": args.num_decode_steps,
        "warmup": args.warmup,
        "max_eval_batches": args.max_eval_batches,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "logit_decode_steps": args.logit_decode_steps,
        "seed": args.seed,
        "ref_prefill_ms": ref_prefill_ms,
        "ref_decode_total_ms": ref_decode_ms,
        "ref_decode_ms_per_token": ms_per_tok_ref,
        "ref_tokens_per_s": tps_ref,
        "ref_peak_cuda_mib": ref_peak,
        "triton_prefill_ms": tr_prefill_ms,
        "triton_decode_total_ms": tr_decode_ms,
        "triton_decode_ms_per_token": ms_per_tok_tr,
        "triton_tokens_per_s": tps_tr,
        "triton_peak_cuda_mib": tr_peak,
        "ppl_reference": ppl_ref,
        "ppl_triton": ppl_tr,
        "rel_ppl_delta_pct": rel_ppl_pct,
        "max_abs_logits_diff": max_logit_diff,
        "mean_abs_logits_diff": mean_logit_diff,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    _write_artifacts(out_dir, metrics)
    return metrics


def main() -> None:
    _ensure_cuda()
    parser = argparse.ArgumentParser(description="GPT-2 reference vs Triton benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help=f"HF model ids to benchmark in order (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="On failure (e.g. CUDA OOM), record error and continue with the next model",
    )
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--num-decode-steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-eval-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--logit-decode-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_ids = args.models if args.models is not None else list(DEFAULT_MODELS)

    device = torch.device("cuda")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = _ROOT / "benchmarks" / "logs" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [row["text"] for row in ds if row.get("text") and row["text"].strip()]
    need = args.batch_size * args.max_eval_batches
    texts = texts[:need]

    cuda_name = torch.cuda.get_device_name(0)

    completed: list[str] = []
    failed: list[dict[str, str]] = []
    cross_rows: list[tuple[str, float, float]] = []

    for model_id in model_ids:
        sub = out_root / model_id
        sub.mkdir(parents=True, exist_ok=True)
        try:
            metrics = run_single_checkpoint(
                model_id,
                sub,
                device=device,
                tokenizer=tokenizer,
                texts=texts,
                run_timestamp=ts,
                cuda_name=cuda_name,
                args=args,
            )
            completed.append(model_id)
            cross_rows.append(
                (
                    model_id,
                    float(metrics["ref_decode_ms_per_token"]),
                    float(metrics["triton_decode_ms_per_token"]),
                )
            )
            print(f"Completed {model_id} -> {sub}", file=sys.stderr)
        except Exception as e:
            err = str(e)
            failed.append({"model_id": model_id, "error": err})
            with open(sub / "error.json", "w", encoding="utf-8") as f:
                json.dump({"model_id": model_id, "error": err}, f, indent=2)
            print(f"FAILED {model_id}: {err}", file=sys.stderr)
            gc.collect()
            torch.cuda.empty_cache()
            if not args.continue_on_error:
                summary = {
                    "timestamp": ts,
                    "cuda_device_name": cuda_name,
                    "models_requested": model_ids,
                    "models_completed": completed,
                    "models_failed": failed,
                }
                with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                raise

    if cross_rows:
        plot_cross_model_decode_latency(out_root / "decode_ms_per_token_by_size.png", cross_rows)

    summary = {
        "timestamp": ts,
        "cuda_device_name": cuda_name,
        "models_requested": model_ids,
        "models_completed": completed,
        "models_failed": failed,
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote artifacts under {out_root}")


if __name__ == "__main__":
    main()
