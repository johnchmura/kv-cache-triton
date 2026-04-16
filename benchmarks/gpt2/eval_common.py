"""Shared helpers for benchmark evals: truncation and greedy generation with KV cache."""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models.gpt2.kv_cache import QuantizedKVCache

PPL_ECE_EVAL_MODE = "teacher_forced_incremental_kv"


def quantized_kv_cache_for_model(model: GPT2LMHeadModel) -> QuantizedKVCache | None:
    attn0 = getattr(getattr(model, "transformer", None), "h", [None])[0]
    quant_group_size = getattr(getattr(attn0, "attn", None), "quant_group_size", None)
    return QuantizedKVCache(group_size=int(quant_group_size)) if quant_group_size is not None else None


@torch.no_grad()
def teacher_forced_incremental_logits(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token logits via KV cache (one forward per position). input_ids (1, L), L >= 1.

    Returns logits (L-1, V) and targets (L-1,) for positions predicting input_ids[:, 1:].
    If L < 2, returns empty tensors.
    """
    model.eval()
    ids = input_ids.to(device)
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError("expected input_ids shape (1, L)")
    L = int(ids.shape[1])
    cfg = getattr(model, "config", None)
    v = int(cfg.vocab_size) if cfg is not None else int(getattr(model, "vocab_size"))
    if L < 2:
        return (
            torch.empty(0, v, device=device, dtype=torch.float32),
            torch.empty(0, dtype=torch.long, device=device),
        )

    quant_cache = quantized_kv_cache_for_model(model)
    logits_chunks: list[torch.Tensor] = []
    past = None

    for t in range(L - 1):
        step_ids = ids[:, t : t + 1]
        if t == 0:
            if quant_cache is None:
                out = model(input_ids=step_ids, use_cache=True)
                past = out.past_key_values
            else:
                out = model(input_ids=step_ids, use_cache=True, past_key_values=quant_cache)
                past = quant_cache
        else:
            out = model(input_ids=step_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
        logits_chunks.append(out.logits[:, -1, :])

    logits = torch.cat(logits_chunks, dim=0).float()
    targets = ids[0, 1:L].long()
    return logits, targets


def effective_max_context(model: GPT2LMHeadModel, max_context_tokens: int | None) -> int:
    cap = int(model.config.n_positions)
    if max_context_tokens is None:
        return cap
    return min(cap, int(max_context_tokens))


def truncate_context_keep_query_tail(
    tokenizer: GPT2Tokenizer,
    context: str,
    query_suffix: str,
    max_tokens: int,
) -> torch.Tensor:
    """Keep the end of context plus full query_suffix; drop oldest context tokens if over budget."""
    enc_q = tokenizer(query_suffix, add_special_tokens=False, return_tensors="pt")
    q_ids = enc_q["input_ids"][0]
    enc_c = tokenizer(context, add_special_tokens=False, return_tensors="pt")
    c_ids = enc_c["input_ids"][0]
    budget = max_tokens
    if q_ids.numel() > budget:
        q_ids = q_ids[-budget:]
        c_ids = c_ids[:0]
    else:
        remain = budget - q_ids.numel()
        if c_ids.numel() > remain:
            c_ids = c_ids[-remain:]
    if c_ids.numel() == 0:
        out = q_ids.unsqueeze(0)
    else:
        out = torch.cat([c_ids, q_ids], dim=0).unsqueeze(0)
    return out.long()


@torch.no_grad()
def greedy_generate_with_cache(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Prefill input_ids (1, seq), then greedy decode max_new_tokens steps. Returns (1, max_new_tokens)."""
    model.eval()
    ids = input_ids.to(device)
    quant_cache = quantized_kv_cache_for_model(model)

    if quant_cache is None:
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
    else:
        out = model(input_ids=ids, use_cache=True, past_key_values=quant_cache)
        past = quant_cache
    logits = out.logits
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_id]

    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id)

    return torch.cat(generated, dim=1)
