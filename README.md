# kv-cache-triton

Triton kernels for attention-related workloads (Phase 1: synthetic tensors; later LLaMA-style quantization hooks).

## Layout

- `kernels/` — Triton kernels (e.g. `attention.py`)
- `tests/` — correctness tests
- `benchmarks/` — GPT-2 reference vs Triton scripts and run logs
- `experiments/` — notebooks or scripts (reserved)
- `models/` — future LLaMA integration (reserved)
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
pytest tests/test_attention.py -v
```

## Smoke run

```bash
python main.py
```

Prints the max absolute difference versus PyTorch SDPA on fixed small shapes.

## Benchmarks

Compares Hugging Face GPT-2 attention with the Triton-patched model (decode steps). Downloads checkpoints, WikiText-2, and (when enabled) LongBench `data.zip` / streaming calibration data on first run (requires network).

```bash
python -m benchmarks.run_gpt2_benchmark
```

By default this runs **gpt2**, **gpt2-medium**, **gpt2-large**, and **gpt2-xl** in order. Use `--models` to limit or reorder, e.g. `--models gpt2 gpt2-large`. Larger checkpoints need more VRAM; if a size fails (for example CUDA OOM), use `--continue-on-error` to finish the remaining models and record failures in `run_summary.json` and per-model `error.json`.

Extra evals (see `--help`): **passkey** retrieval (optional `--ruler-style` two-key haystack), **LongBench** subsets loaded from `THUDM/LongBench` `data.zip` with tail-context truncation (not comparable to full-context leaderboards; GPT-2 is 1024 tokens), and **calibration** token ECE on a small C4 or Pile stream. Use `--passkey-samples 0`, `--longbench-max-samples 0`, or `--calibration-samples 0` to disable each.

Each run writes a timestamped root folder `benchmarks/logs/<YYYY-MM-DD_HHMMSS>/` containing:

- `run_summary.json` — which models completed or failed
- `decode_ms_per_token_by_size.png` — cross-model decode latency (when at least one model completed)
- Per checkpoint subdirectory `<model_id>/` with `results.csv`, `metrics.json`, and PNGs: `decode_latency.png`, `peak_vram.png`, `throughput.png`, `perplexity.png`, `logit_agreement.png`, plus when applicable `passkey_match.png`, `longbench_substring_hit.png`, `calibration_ece.png`

CUDA is required.
