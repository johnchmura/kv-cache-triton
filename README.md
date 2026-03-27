# kv-cache-triton

Triton kernels for attention-related workloads (Phase 1: synthetic tensors; later LLaMA-style quantization hooks).

## Layout

- `kernels/` — Triton kernels (e.g. `attention.py`)
- `tests/` — correctness tests
- `benchmarks/` — latency and memory experiments (reserved)
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
