"""Shared helpers for benchmark evals: truncation and greedy generation with KV cache."""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_id]

    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id)

    return torch.cat(generated, dim=1)
