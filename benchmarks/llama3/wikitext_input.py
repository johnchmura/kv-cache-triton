"""Shared WikiText-2 validation row picker for Llama debug/benchmark scripts."""

from __future__ import annotations

from typing import Any


def wikitext_enc_first_row(
    tokenizer: Any,
    max_length: int,
    *,
    min_tokens: int = 2,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Pick a WikiText-2 validation row and tokenize with ``max_length`` truncation.

    Default behaviour (``row_index=None``): return the first non-empty row whose
    tokenized length is in ``[min_tokens, max_length]``.

    When ``row_index`` is set, return that specific non-empty row (0-indexed over
    non-empty rows), ignoring ``min_tokens``. Use this for length ablations so
    changing ``max_length`` does not select a different prompt.
    """
    from datasets import load_dataset

    if min_tokens < 2:
        min_tokens = 2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    if row_index is not None:
        if row_index < 0:
            raise ValueError("row_index must be >= 0")
        seen = 0
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            if seen == row_index:
                return tokenizer(
                    text, truncation=True, max_length=max_length, return_tensors="pt"
                )
            seen += 1
        raise RuntimeError(
            f"wikitext validation has only {seen} non-empty rows; row_index={row_index} out of range"
        )
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        n = int(enc["input_ids"].shape[1])
        if n >= min_tokens:
            return enc
    raise RuntimeError(
        f"no wikitext validation row with length >= {min_tokens} (max_length={max_length})"
    )
