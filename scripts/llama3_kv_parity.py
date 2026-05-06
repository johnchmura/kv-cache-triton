"""Greedy parity: Llama 3.1 8B baseline (BF16 KV) vs quantized (INT4 KV).

Runs greedy decode on both paths over the same prompt and logs:
    - max |logits diff| per decode step
    - top-1 agreement per decode step
    - first divergence token index

Usage:
    python scripts/llama3_kv_parity.py
    python scripts/llama3_kv_parity.py --max-new-tokens 64 --group-size 32
    python scripts/llama3_kv_parity.py --kernel-variant sm70
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.llama3.kernel_variant import bind_quantized_kernel_variant, resolve_kernel_variant

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. The capital of France is"


def _load(model_id: str, quantized: bool, group_size: int, replace_fn):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    if quantized:
        replace_fn(model, group_size=group_size)
    return model


def _embed_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda", 0)


def _greedy_step(model, input_ids: torch.Tensor, past) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    return out.logits[:, -1, :], past


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--kernel-variant",
        choices=("auto", "default", "sm70"),
        default="auto",
        help="INT4 kernel path: auto uses sm70 on GPUs below sm_80 (same as run_llama3_benchmark).",
    )
    parser.add_argument("--out-json", default=None, help="Optional JSON dump of per-step metrics")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; parity requires a GPU.")

    variant = resolve_kernel_variant(args.kernel_variant)
    QuantizedKVCache, replace_fn = bind_quantized_kernel_variant(variant)
    print(f"[info] kernel variant: {variant}")

    n_gpu = max(1, torch.cuda.device_count())
    gpu_names = {torch.cuda.get_device_name(d) for d in range(n_gpu)}
    print(f"[info] device: {n_gpu}x {'/'.join(sorted(gpu_names))}")
    print(f"[info] loading baseline (BF16 KV)...")
    t0 = time.perf_counter()
    model_ref = _load(args.model, quantized=False, group_size=args.group_size, replace_fn=replace_fn)
    ref_load = time.perf_counter() - t0
    print(f"[info] baseline loaded in {ref_load:.1f}s")

    tok = AutoTokenizer.from_pretrained(args.model)
    embed_dev = _embed_device(model_ref)
    inputs = tok(args.prompt, return_tensors="pt").to(embed_dev)
    prompt_ids = inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    print(f"[info] prompt tokens: {prompt_len}")

    print(f"[info] prefilling baseline...")
    cache_ref = DynamicCache()
    logits_ref, cache_ref = _greedy_step(model_ref, prompt_ids, cache_ref)
    tokens_ref: list[int] = [int(logits_ref.argmax(-1).item())]
    logits_ref_per_step: list[torch.Tensor] = [logits_ref.detach().cpu()]
    next_ref = torch.tensor([[tokens_ref[-1]]], device=embed_dev, dtype=torch.long)
    for _ in range(args.max_new_tokens - 1):
        logits_ref, cache_ref = _greedy_step(model_ref, next_ref, cache_ref)
        tokens_ref.append(int(logits_ref.argmax(-1).item()))
        logits_ref_per_step.append(logits_ref.detach().cpu())
        next_ref = torch.tensor([[tokens_ref[-1]]], device=embed_dev, dtype=torch.long)

    del model_ref
    torch.cuda.empty_cache()

    print(f"[info] loading quantized (INT4 KV, group_size={args.group_size})...")
    t0 = time.perf_counter()
    model_q = _load(args.model, quantized=True, group_size=args.group_size, replace_fn=replace_fn)
    q_load = time.perf_counter() - t0
    print(f"[info] quantized loaded in {q_load:.1f}s")
    embed_dev_q = _embed_device(model_q)
    prompt_ids = prompt_ids.to(embed_dev_q)

    print(f"[info] prefilling quantized...")
    cache_q = QuantizedKVCache(group_size=args.group_size)
    logits_q, cache_q = _greedy_step(model_q, prompt_ids, cache_q)
    tokens_q: list[int] = [int(logits_q.argmax(-1).item())]
    logits_q_per_step: list[torch.Tensor] = [logits_q.detach().cpu()]
    next_q = torch.tensor([[tokens_q[-1]]], device=embed_dev_q, dtype=torch.long)
    for _ in range(args.max_new_tokens - 1):
        logits_q, cache_q = _greedy_step(model_q, next_q, cache_q)
        tokens_q.append(int(logits_q.argmax(-1).item()))
        logits_q_per_step.append(logits_q.detach().cpu())
        next_q = torch.tensor([[tokens_q[-1]]], device=embed_dev_q, dtype=torch.long)

    quant_kv_mib = cache_q.nbytes() / 1024**2
    print(f"[info] quant KV storage after {args.max_new_tokens} steps: {quant_kv_mib:.2f} MiB")

    per_step: list[dict] = []
    first_div = None
    for i, (a, b) in enumerate(zip(logits_ref_per_step, logits_q_per_step)):
        diff = (a.float() - b.float()).abs().max().item()
        top1_ref = int(a.argmax(-1).item())
        top1_q = int(b.argmax(-1).item())
        agree = top1_ref == top1_q
        per_step.append(
            {"step": i, "max_abs_logit_diff": diff, "top1_ref": top1_ref, "top1_quant": top1_q, "top1_agree": agree}
        )
        if first_div is None and tokens_ref[i] != tokens_q[i]:
            first_div = i

    print("step  max|diff|  top1_ref  top1_quant  agree")
    for row in per_step:
        print(
            f"{row['step']:>4}  {row['max_abs_logit_diff']:9.4f}  "
            f"{row['top1_ref']:>8}  {row['top1_quant']:>10}  {row['top1_agree']}"
        )

    print(f"[summary] first_divergence_step: {first_div}")
    print(f"[summary] ref tokens ({len(tokens_ref)}): {tokens_ref}")
    print(f"[summary] quant tokens ({len(tokens_q)}): {tokens_q}")
    print(f"[summary] ref text:   {tok.decode(prompt_ids[0].tolist() + tokens_ref, skip_special_tokens=True)}")
    print(f"[summary] quant text: {tok.decode(prompt_ids[0].tolist() + tokens_q, skip_special_tokens=True)}")

    if args.out_json is not None:
        data = {
            "model": args.model,
            "kernel_variant": variant,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "prompt": args.prompt,
            "quant_kv_mib": quant_kv_mib,
            "first_divergence_step": first_div,
            "per_step": per_step,
            "tokens_ref": tokens_ref,
            "tokens_quant": tokens_q,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[info] wrote {args.out_json}")


if __name__ == "__main__":
    main()
