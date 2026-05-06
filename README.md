# kv-cache-triton

Triton kernels for attention-related workloads (Phase 1: synthetic tensors; later LLaMA-style quantization hooks).

## Layout

- `kernels/` — Triton kernels (GPT-2 lives in `kernels/gpt2/`)
- `tests/` — test suites grouped by model family (GPT-2 lives in `tests/gpt2/`)
- `benchmarks/` — benchmark suites grouped by model family (GPT-2 lives in `benchmarks/gpt2/`)
- `experiments/` — notebooks or scripts (reserved)
- `models/` — model glue code grouped by model family (GPT-2 lives in `models/gpt2/`)
- `utils/` — shared helpers (reserved)

## Phase 1

Single-query scaled dot-product attention: `Q` is `[batch, heads, 1, d_head]`, `K` and `V` are `[batch, heads, seq_len, d_head]`. Outputs match `torch.nn.functional.scaled_dot_product_attention` on random data (no causal mask).

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA for the kernel and tests.

## Run tests

```bash
pytest tests/gpt2/test_attention.py -v
```

## Smoke run

```bash
python main.py
```

Prints the max absolute difference versus PyTorch SDPA on fixed small shapes.

## Benchmarks

Compares Hugging Face GPT-2 reference attention with the quantized KV cache / attention path (decode steps, storage, and optional evals). Downloads checkpoints, WikiText-2, and (when enabled) LongBench `data.zip` / streaming calibration data on first run (requires network).

```bash
python -m benchmarks.gpt2.run_gpt2_benchmark
```

By default this runs **gpt2**, **gpt2-medium**, **gpt2-large**, and **gpt2-xl** in order. Use `--models` to limit or reorder, e.g. `--models gpt2 gpt2-large`. Larger checkpoints need more VRAM; if a size fails (for example CUDA OOM), use `--continue-on-error` to finish the remaining models and record failures in `run_summary.json` and per-model `error.json`.

Extra evals (see `--help`): **passkey** retrieval (optional `--ruler-style` two-key haystack), **LongBench** subsets loaded from `THUDM/LongBench` `data.zip` with tail-context truncation (not comparable to full-context leaderboards; GPT-2 is 1024 tokens), and **calibration** token ECE on a small C4 or Pile stream. Use `--passkey-samples 0`, `--longbench-max-samples 0`, or `--calibration-samples 0` to disable each.

Each run writes a timestamped root folder `benchmarks/logs/<YYYY-MM-DD_HHMMSS>/` containing:

- `run_summary.json` — which models completed or failed
- `decode_ms_per_token_by_size.png` — cross-model decode latency (when at least one model completed)
- Per checkpoint subdirectory `<model_id>/` with `results.csv`, `metrics.json`, and PNGs: `decode_latency.png`, `peak_vram.png`, `throughput.png`, `perplexity.png`, `logit_agreement.png`, plus when applicable `passkey_match.png`, `longbench_substring_hit.png`, `calibration_ece.png`

CUDA is required.

## Llama 3.1 benchmarks (8B + 70B)

Compares the stock PyTorch `DynamicCache` (BF16 KV) against `QuantizedKVCache` (fused INT4 KV via Triton) on Llama 3.1 base checkpoints. Default sweep is `meta-llama/Llama-3.1-8B` then `meta-llama/Llama-3.1-70B`. The runner loads with `device_map="auto"` and lets Hugging Face place weights across available GPUs. The 70B pass requires a multi-GPU host (e.g. 8x32GB).

On pre-Ampere GPUs (V100, sm_70) the default Triton kernels fail to build because they emit `.bf16` PTX instructions that require sm_80+. The runner auto-detects this and switches to the sm_70-safe copies under [`kernels/llama3/attention_quant_sm70.py`](kernels/llama3/attention_quant_sm70.py), [`kernels/llama3/quantize_sm70.py`](kernels/llama3/quantize_sm70.py), [`models/llama3/kv_cache_sm70.py`](models/llama3/kv_cache_sm70.py), and [`models/llama3/llama3_quant_sm70.py`](models/llama3/llama3_quant_sm70.py). Pass `--kernel-variant {auto,default,sm70}` to force a choice; the default is `auto`.

### One-time remote setup (70B weights)

The 70B weights are ~140 GB. Pre-fetch them on the remote host so they never transit your laptop. From s8:

```bash
export HF_TOKEN=hf_...              # must have access to meta-llama repos
bash scripts/setup_llama_remote.sh  # writes to $HF_HOME (default /ceph/jchmura/hf_cache)
```

Or push-and-run from your laptop without checking out the repo on s8:

```bash
ssh -J mystic1 s8 "HF_TOKEN=hf_... bash -s" < scripts/setup_llama_remote.sh
```

Override which models to fetch via `MODELS="meta-llama/Llama-3.1-70B"` and the cache dir via `HF_HOME=/path/to/hf_cache`.

### Running the sweep

After syncing the repo (`./scripts/rsync_s8.sh`) and setting `HF_HOME` on s8:

```bash
export HF_HOME=/ceph/jchmura/hf_cache
python -m benchmarks.llama3.run_llama3_benchmark --continue-on-error
```

Useful flags: `--models <ids...>`, `--prefill-lens 1024 4096 16384`, `--num-decode-steps 64`, `--group-size 32`, `--ppl-samples 32`, `--passkey-samples 0`, `--longbench-subsets narrativeqa,triviaqa`, `--kernel-variant {auto,default,sm70}`.

V100 / pre-Ampere example (auto-selects `sm70` kernels):

```bash
python -m benchmarks.llama3.run_llama3_benchmark \
  --models meta-llama/Llama-3.1-8B \
  --prefill-lens 512 1024 2048 4096 \
  --continue-on-error
```

Output layout under `benchmarks/llama3/logs/<ts>/`:

- `run_summary.json` plus cross-model PNGs (`decode_ms_per_token_by_size.png`, `kv_cache_storage_mib_by_size.png`, `peak_vram_mib_by_size.png`).
- `Llama-3.1-8B/` and `Llama-3.1-70B/` subdirectories each with `model_summary.json`, per-length `prefill_<L>/{metrics.json, results.csv, decode_latency.png, peak_vram.png, kv_cache_storage_mib.png}`, plus `decode_ms_per_token_by_seqlen.png`, `kv_cache_storage_mib_by_seqlen.png`, `peak_vram_mib_by_seqlen.png`.
- If a model errors, `<model>/error.json` is written and (with `--continue-on-error`) the sweep moves on.

Peak VRAM is reported as the sum across all visible GPUs, with per-device breakdown in the JSON.

### Multi-GPU lifecycle

For each checkpoint the runner loads the weights once, runs the reference (`DynamicCache`) pass, then swaps every decoder-layer attention module to `LlamaAttentionQuantized` in place (re-using the existing q/k/v/o parameters via `_transplant_projection`) and runs the quantized pass. This keeps peak memory at ~one model copy plus KV, so 70B fits on 8x32GB with headroom.

### Other Llama 3.1 scripts

- `python scripts/llama3_smoke.py --model meta-llama/Llama-3.1-70B --quant-kv` — end-to-end generate smoke test.
- `python scripts/llama3_kv_parity.py --model meta-llama/Llama-3.1-70B` — logits-level parity check between the two paths.
- `python -m benchmarks.llama3.microbench_kernel` — kernel-only INT4 vs SDPA microbenchmark (no model download required).
- `python -m benchmarks.llama3.group_size_sweep` — INT4 group-size ablation.
