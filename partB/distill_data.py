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
    "english": "Reason step by step in English and respond",
    "hindi":   "Reason step by step in English and respond",
    "bengali": "Reason step by step in English and respond",
    "kannada": "Reason step by step in English and respond",
    "tamil":   "Reason step by step in English and respond",
}

CORRECT_WEIGHT   = 1.0
INCORRECT_WEIGHT = 0.5

INCORRECT_FRACTION_RATIO = 0.0
VAL_FRACTION = 0.05

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

def load_and_split(jsonl_path, tokenizer, val_fraction=VAL_FRACTION, seed=42):
    import json, random
    from collections import defaultdict

    all_raw = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_raw.append(json.loads(line))

    correct   = [r for r in all_raw if r.get("correct", False)]
    incorrect = [r for r in all_raw if not r.get("correct", False)]

    LOGGER.info("Loaded: %d correct, %d incorrect", len(correct), len(incorrect))

    # 15% incorrect per language
    by_lang_incorrect = defaultdict(list)
    for r in incorrect:
        lang = _canonical_language(r.get("language", "en"))
        by_lang_incorrect[lang].append(r)

    random.seed(seed)
    kept_incorrect = []
    for lang, rows in by_lang_incorrect.items():
        n_keep = max(1, int(len(rows) * INCORRECT_FRACTION_RATIO))
        random.shuffle(rows)
        kept_incorrect.extend(rows[:n_keep])

    final_raw = correct + kept_incorrect
    random.shuffle(final_raw)

    LOGGER.info(
        "Final: %d correct + %d incorrect (0%%) = %d total",
        len(correct), len(kept_incorrect), len(final_raw)
    )

    # Log per-language breakdown
    by_lang = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for r in final_raw:
        lang = _canonical_language(r.get("language", "en"))
        key = "correct" if r.get("correct", False) else "incorrect"
        by_lang[lang][key] += 1
    for lang, counts in sorted(by_lang.items()):
        LOGGER.info("  %-10s : %d correct + %d incorrect",
                    lang, counts["correct"], counts["incorrect"])

    # Convert to DistillRecord — build prompt here
    def to_record(r: dict) -> DistillRecord:
        lang     = _canonical_language(r.get("language", "en"))
        prompt   = build_prompt(r["question"], lang, tokenizer)
        generation = r.get("teacher_generation", "")
        weight   = CORRECT_WEIGHT if r.get("correct", False) else INCORRECT_WEIGHT
        return DistillRecord(
            prompt=prompt,
            generation=generation,
            weight=weight,
            gold_answer=str(r.get("gold_answer", r.get("answer", ""))).upper()[:1],
            language=lang,
        )

    records = [to_record(r) for r in final_raw]

    # Val split
    split_idx     = max(1, int(len(records) * val_fraction))
    val_records   = records[:split_idx]
    train_records = records[split_idx:]

    LOGGER.info("Train: %d | Val: %d", len(train_records), len(val_records))
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