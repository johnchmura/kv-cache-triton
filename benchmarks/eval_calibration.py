"""LM calibration (ECE) on a small C4 or Pile subset."""

from __future__ import annotations

from typing import Literal

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmarks.eval_common import teacher_forced_incremental_logits


def load_calibration_texts(
    source: Literal["c4", "pile"],
    n_samples: int,
    seed: int,
) -> list[str]:
    texts: list[str] = []
    if source == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        stream = ds.shuffle(seed=seed, buffer_size=10_000)
        for row in stream:
            t = row.get("text", "")
            if t and str(t).strip():
                texts.append(str(t))
            if len(texts) >= n_samples:
                break
        return texts

    stream = None
    for ds_id, split in (
        ("EleutherAI/pile", "train"),
        ("JeanKaddour/minipile", "train"),
    ):
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
            stream = ds.shuffle(seed=seed, buffer_size=10_000)
            break
        except Exception:
            continue
    if stream is None:
        raise RuntimeError(
            "Could not load a Pile streaming dataset (try --calibration-source c4)."
        )
    for row in stream:
        t = row.get("text", "")
        if t and str(t).strip():
            texts.append(str(t))
        if len(texts) >= n_samples:
            break
    return texts


def compute_lm_ece(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    num_bins: int = 15,
) -> tuple[float, int]:
    """Token-level ECE: bins on p(true next token); accuracy = 1 if argmax equals target."""
    model.eval()
    confidences: list[float] = []
    accuracies: list[float] = []

    for text in texts:
        if not str(text).strip():
            continue
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logit_pred, targets = teacher_forced_incremental_logits(model, input_ids, device)
        if logit_pred.shape[0] == 0:
            continue
        probs = torch.softmax(logit_pred, dim=-1)
        true_prob = probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        pred_max = logit_pred.argmax(dim=-1)
        acc = (pred_max == targets).float()
        confidences.extend(true_prob.cpu().tolist())
        accuracies.extend(acc.cpu().tolist())

    if not confidences:
        return float("nan"), 0

    conf_t = torch.tensor(confidences)
    acc_t = torch.tensor(accuracies)
    n = len(confidences)
    edges = torch.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        low, high = edges[i].item(), edges[i + 1].item()
        last = i == num_bins - 1
        if last:
            mask = (conf_t >= low) & (conf_t <= high)
        else:
            mask = (conf_t >= low) & (conf_t < high)
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        conf_bin = conf_t[mask].mean().item()
        acc_bin = acc_t[mask].mean().item()
        ece += (cnt / n) * abs(acc_bin - conf_bin)

    return float(ece), n
