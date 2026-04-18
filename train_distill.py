from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

LOGGER = logging.getLogger(__name__)

# Constants                                                            #
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LANGUAGE_INSTRUCTIONS = {
    "en":      "Respond in English.",
    "hindi":   "अपना उत्तर हिंदी में दें।",
    "bengali": "আপনার উত্তর বাংলায় দিন।",
    "kannada": "Reason step by step in English but present your final answer line in Kannada.",
    "tamil":   "Reason step by step in English but present your final answer line in Tamil.",
}

CORRECT_WEIGHT   = 1.0
INCORRECT_WEIGHT = 0.5

ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)

# Max val samples per language for accuracy eval (keep it fast)
VAL_ACC_SAMPLES_PER_LANG = 50

# Data structures                                                      #
@dataclass
class DistillRecord:
    prompt: str
    generation: str
    weight: float
    gold_answer: str
    language: str

# Prompt builder                                                       #
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system_content}\nUser: {question}\nAssistant:"


# Data loading + stratified split                                      #
def _canonical_language(lang: str) -> str:
    aliases = {
        "en": "en", "eng": "en", "english": "en",
        "hi": "hindi", "hindi": "hindi",
        "bn": "bengali", "bengali": "bengali",
        "kn": "kannada", "kannada": "kannada",
        "ta": "tamil", "tamil": "tamil",
    }
    return aliases.get(str(lang).strip().lower(), "en")


def load_and_split(
    train_data: str,
    tokenizer,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[DistillRecord], list[DistillRecord]]:
    """Load JSONL, build prompts, then do language-stratified 90/10 split."""

    # Group raw rows by language first
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
        n_val = max(1, int(len(rows) * val_fraction))
        val_rows   = rows[:n_val]
        train_rows = rows[n_val:]

        LOGGER.info(
            "  %-10s : %d train | %d val",
            lang, len(train_rows), len(val_rows),
        )

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


# Dataset + collate                                                    #
class DistillDataset(Dataset):
    def __init__(self, records: list[DistillRecord], tokenizer, max_length: int = 1536):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        full_text = rec.prompt + rec.generation

        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            rec.prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
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
        labels_list.append(F.pad(item["labels"],    (0, pad_len), value=-100))
        weights_list.append(item["weight"])

    stacked_ids = torch.stack(input_ids_list)
    return {
        "input_ids":      stacked_ids,
        "attention_mask": (stacked_ids != pad_token_id).long(),
        "labels":         torch.stack(labels_list),
        "weights":        torch.stack(weights_list),
    }

# Loss                                                                 #
def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels = labels[:, 1:].contiguous().view(-1)

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none"
    ).view(B, T - 1)

    valid_mask   = (labels[:, 1:] != -100).float()
    token_counts = valid_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / token_counts

    return (per_sample_loss * weights).mean()

# Validation                                                           #
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
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(total_steps, 1)

@torch.no_grad()
def run_val_accuracy(model, tokenizer, val_records, device,
                     max_new_tokens=256, samples_per_lang=VAL_ACC_SAMPLES_PER_LANG,
                     seed=42, gen_batch_size=8) -> dict[str, float]:
    model.eval()

    # --- sample per language (same as before) ---
    lang_buckets = defaultdict(list)
    for rec in val_records:
        lang_buckets[rec.language].append(rec)
    rng = random.Random(seed)
    subset = []
    for lang, recs in lang_buckets.items():
        sample = recs[:samples_per_lang] if len(recs) <= samples_per_lang \
                 else rng.sample(recs, samples_per_lang)
        subset.extend(sample)

    lang_correct: dict[str, int] = defaultdict(int)
    lang_total:   dict[str, int] = defaultdict(int)

    # --- process in batches ---
    for batch_start in tqdm(range(0, len(subset), gen_batch_size),
                            desc="Val accuracy", leave=False):
        batch_recs = subset[batch_start: batch_start + gen_batch_size]

        # tokenize all prompts together, left-pad so generated tokens align
        encodings = tokenizer(
            [r.prompt for r in batch_recs],
            return_tensors="pt",
            truncation=True,
            max_length=1280,
            padding=True,          # pads shorter prompts on the LEFT
            padding_side="left",   # required for generation
        ).to(device)

        out = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # decode only the newly generated tokens for each item
        prompt_lens = encodings["attention_mask"].sum(dim=1)  # actual (non-pad) lengths
        for i, rec in enumerate(batch_recs):
            gen_ids = out[i][prompt_lens[i]:]          # strip prompt tokens
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

            match = ANSWER_RE.search(generated)
            predicted = match.group(1).upper() if match else ""
            if not predicted:
                letters = re.findall(r'\b([A-J])\b', generated, re.IGNORECASE)
                predicted = letters[-1].upper() if letters else ""

            lang_total[rec.language]   += 1
            lang_correct[rec.language] += int(predicted == rec.gold_answer)

    # --- logging (same as before) ---
    results = {}
    for lang in lang_total:
        acc = lang_correct[lang] / lang_total[lang]
        results[lang] = acc
        LOGGER.info("  Val acc %-10s : %.1f%% (%d/%d)",
                    lang, acc * 100, lang_correct[lang], lang_total[lang])
    overall = sum(lang_correct.values()) / max(sum(lang_total.values()), 1)
    results["overall"] = overall
    LOGGER.info("  Val acc overall    : %.1f%%", overall * 100)
    return results

# Training loop                                                        #
def train(args: argparse.Namespace) -> None:
    LOGGER.info("Loading student tokenizer from %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # Load + split data
    train_records, val_records = load_and_split(
        args.train_data, tokenizer,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    pad_id = tokenizer.pad_token_id
    train_ds  = DistillDataset(train_records, tokenizer, args.max_length)
    val_ds    = DistillDataset(val_records,   tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0, pin_memory=True,
    )

    LOGGER.info("Loading student model from %s", args.student_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = (len(train_loader) // args.grad_accumulation_steps) * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    LOGGER.info("Training: %d epochs | %d steps | warmup %d", args.epochs, total_steps, warmup_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            weights        = batch["weights"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = weighted_cross_entropy(outputs.logits, labels, weights)
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * args.grad_accumulation_steps

            if step % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "train_loss": f"{epoch_loss / step:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_train_loss = epoch_loss / len(train_loader)
        LOGGER.info("Epoch %d | train loss: %.4f", epoch, avg_train_loss)

        # ---- Val loss ----
        val_loss = run_val_loss(model, val_loader, device)
        LOGGER.info("Epoch %d | val loss:   %.4f", epoch, val_loss)

        # ---- Val accuracy (every epoch, capped subset) ----
        LOGGER.info("Epoch %d | running val accuracy...", epoch)
        run_val_accuracy(
            model, tokenizer, val_records, device,
            max_new_tokens=args.val_max_new_tokens,
            samples_per_lang=args.val_acc_samples_per_lang,
            seed=args.seed,
        )

        # ---- Save best by val loss ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            LOGGER.info("Epoch %d | saved best checkpoint (val_loss=%.4f)", epoch, val_loss)

        model.train()

    # Save final
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    LOGGER.info("Saved final checkpoint to %s", final_path)

# Args + entry point                                                   #
def setup_logger(level: str, log_file_name: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"output_{log_file_name}.txt")

    root = logging.getLogger()
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.setLevel(numeric_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    LOGGER.info("Logging to %s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT distillation training")
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_model", required=False,
                        help="Unused in offline distillation, kept for CLI compatibility")
    parser.add_argument("--train_data", default="outputs/train.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of each language's data to use for validation")
    parser.add_argument("--val_max_new_tokens", type=int, default=256,
                        help="Max tokens to generate during val accuracy eval")
    parser.add_argument("--val_acc_samples_per_lang", type=int,
                        default=VAL_ACC_SAMPLES_PER_LANG,
                        help="Max val samples per language for accuracy eval")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file_name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level, args.log_file_name)
    train(args)


if __name__ == "__main__":
    main()