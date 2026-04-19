from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

LANGUAGE_INSTRUCTIONS = {
    "en":      "Respond in English.",
    "hindi":   "अपना उत्तर हिंदी में दें।",
    "bengali": "আপনার উত্তর বাংলায় দিন।",
    "kannada": "Reason step by step in English but present your final answer line in Kannada.",
    "tamil":   "Reason step by step in English but present your final answer line in Tamil.",
}

CORRECT_WEIGHT   = 1.0
INCORRECT_WEIGHT = 0.5


@dataclass
class DistillRecord:
    prompt:      str
    generation:  str
    weight:      float
    gold_answer: str
    language:    str


def _canonical_language(lang: str) -> str:
    aliases = {
        "en": "en", "eng": "en", "english": "en",
        "hi": "hindi", "hindi": "hindi",
        "bn": "bengali", "bengali": "bengali",
        "kn": "kannada", "kannada": "kannada",
        "ta": "tamil", "tamil": "tamil",
    }
    return aliases.get(str(lang).strip().lower(), "en")


def build_prompt(question: str, language: str, tokenizer) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, "Respond in English.")
    system_content = (
        "You are an expert reasoning assistant. "
        f"{lang_instruction} "
        "Think step by step. "
        "Wrap ALL of your reasoning inside <reasoning>...</reasoning> tags. "
        "After the closing tag, end your response with exactly: #### ANSWER: [LETTER] "
        "where [LETTER] is one of A-J. Do not add anything after the answer line."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"System: {system_content}\nUser: {question}\nAssistant:"


def load_and_split(
    train_data: str,
    tokenizer,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[DistillRecord], list[DistillRecord]]:
    """Load JSONL and do a language-stratified train/val split."""
    lang_buckets: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    with open(train_data, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("teacher_generation", "").strip():
                skipped += 1
                continue
            lang = _canonical_language(row.get("language", "en"))
            lang_buckets[lang].append(row)

    LOGGER.info("Loaded rows per language (before split):")
    for lang, rows in sorted(lang_buckets.items()):
        LOGGER.info("  %-10s : %d rows", lang, len(rows))
    if skipped:
        LOGGER.info("Skipped %d rows with empty teacher_generation", skipped)

    rng = random.Random(seed)
    train_records: list[DistillRecord] = []
    val_records:   list[DistillRecord] = []

    for lang, rows in lang_buckets.items():
        rng.shuffle(rows)
        n_val      = max(1, int(len(rows) * val_fraction))
        val_rows   = rows[:n_val]
        train_rows = rows[n_val:]
        LOGGER.info("  %-10s : %d train | %d val", lang, len(train_rows), len(val_rows))

        for split_rows, record_list in [(train_rows, train_records), (val_rows, val_records)]:
            for row in split_rows:
                prompt = build_prompt(
                    question=row["question"],
                    language=lang,
                    tokenizer=tokenizer,
                )
                weight = CORRECT_WEIGHT if row.get("correct", True) else INCORRECT_WEIGHT
                gold   = str(row.get("gold_answer", "")).upper()[:1]
                record_list.append(DistillRecord(
                    prompt=prompt,
                    generation=row["teacher_generation"].strip(),
                    weight=weight,
                    gold_answer=gold,
                    language=lang,
                ))

    LOGGER.info("Final: %d train | %d val records", len(train_records), len(val_records))
    return train_records, val_records


class DistillDataset(Dataset):
    def __init__(self, records: list[DistillRecord], tokenizer, max_length: int = 1536):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec       = self.records[idx]
        full_text = rec.prompt + rec.generation

        full_enc = self.tokenizer(
            full_text, max_length=self.max_length,
            truncation=True, padding=False, return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            rec.prompt, max_length=self.max_length,
            truncation=True, padding=False, return_tensors="pt",
        )

        input_ids  = full_enc["input_ids"].squeeze(0)
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if prompt_len >= len(labels):
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "labels":    labels,
            "weight":    torch.tensor(rec.weight, dtype=torch.float32),
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list, labels_list, weights_list = [], [], []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids_list.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels_list.append(F.pad(item["labels"],       (0, pad_len), value=-100))
        weights_list.append(item["weight"])

    stacked_ids = torch.stack(input_ids_list)
    return {
        "input_ids":      stacked_ids,
        "attention_mask": (stacked_ids != pad_token_id).long(),
        "labels":         torch.stack(labels_list),
        "weights":        torch.stack(weights_list),
    }