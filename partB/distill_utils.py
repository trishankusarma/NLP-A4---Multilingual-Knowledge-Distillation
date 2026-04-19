from __future__ import annotations

import logging
import random
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .distill_data import DistillRecord

LOGGER = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
VAL_ACC_SAMPLES_PER_LANG = 50


def weighted_cross_entropy(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)
    weights: torch.Tensor,  # (B,)
) -> torch.Tensor:
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels = labels[:, 1:].contiguous().view(-1)

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none"
    ).view(B, T - 1)

    valid_mask      = (labels[:, 1:] != -100).float()
    token_counts    = valid_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / token_counts

    return (per_sample_loss * weights).mean()


@torch.no_grad()
def run_val_loss(
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss, total_steps = 0.0, 0

    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        weights        = batch["weights"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = weighted_cross_entropy(outputs.logits, labels, weights)
        total_loss  += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def run_val_accuracy(
    model,
    tokenizer,
    val_records: list[DistillRecord],
    device: torch.device,
    max_new_tokens: int = 512,
    samples_per_lang: int = VAL_ACC_SAMPLES_PER_LANG,
    gen_batch_size: int = 8,
    seed: int = 42,
) -> dict[str, float]:
    """Generate answers on a capped subset, report per-language accuracy."""
    model.eval()

    # Sample per language
    lang_buckets: dict[str, list[DistillRecord]] = defaultdict(list)
    for rec in val_records:
        lang_buckets[rec.language].append(rec)

    rng = random.Random(seed)
    subset: list[DistillRecord] = []
    for lang, recs in lang_buckets.items():
        sample = recs[:samples_per_lang] if len(recs) <= samples_per_lang \
                 else rng.sample(recs, samples_per_lang)
        subset.extend(sample)

    lang_correct: dict[str, int] = defaultdict(int)
    lang_total:   dict[str, int] = defaultdict(int)

    for batch_start in tqdm(range(0, len(subset), gen_batch_size),
                            desc="Val accuracy", leave=False):
        batch_recs = subset[batch_start: batch_start + gen_batch_size]

        encodings = tokenizer(
            [r.prompt for r in batch_recs],
            return_tensors="pt",
            truncation=True,
            max_length=1280,
            padding=True,
            padding_side="left",
        ).to(device)

        out = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_lens = encodings["attention_mask"].sum(dim=1)
        for i, rec in enumerate(batch_recs):
            gen_ids   = out[i][prompt_lens[i]:]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

            match = ANSWER_RE.search(generated)
            if match:
                predicted = match.group(1).upper()
            else:
                letters   = re.findall(r'\b([A-J])\b', generated, re.IGNORECASE)
                predicted = letters[-1].upper() if letters else ""

            lang_total[rec.language]   += 1
            lang_correct[rec.language] += int(predicted == rec.gold_answer)

    results: dict[str, float] = {}
    for lang in sorted(lang_total):
        acc = lang_correct[lang] / lang_total[lang]
        results[lang] = acc
        LOGGER.info("  Val acc %-10s : %.1f%% (%d/%d)",
                    lang, acc * 100, lang_correct[lang], lang_total[lang])

    overall = sum(lang_correct.values()) / max(sum(lang_total.values()), 1)
    results["overall"] = overall
    LOGGER.info("  Val acc overall    : %.1f%%", overall * 100)
    return results