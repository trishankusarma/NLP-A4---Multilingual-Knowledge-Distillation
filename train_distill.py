from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from partB.distill_data import DistillDataset, collate_fn, load_and_split
from partB.distill_utils import VAL_ACC_SAMPLES_PER_LANG, run_val_accuracy, run_val_loss, weighted_cross_entropy

LOGGER = logging.getLogger(__name__)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training loop
def train(args: argparse.Namespace) -> None:
    LOGGER.info("Loading student tokenizer from %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # ---- Data -----
    train_records, val_records = load_and_split(
        args.train_data, tokenizer,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    pad_id   = tokenizer.pad_token_id
    train_ds = DistillDataset(train_records, tokenizer, args.max_length)
    val_ds   = DistillDataset(val_records,   tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0, pin_memory=True,
    )

    # ---- Model + LoRA ----
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
    model  = model.to(device)

    # ---- Optimizer + scheduler ----
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = (len(train_loader) // args.grad_accumulation_steps) * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    LOGGER.info("Training: %d epochs | %d steps | warmup %d", args.epochs, total_steps, warmup_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Early stopping state ----
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    # Epoch loop
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
            loss    = weighted_cross_entropy(outputs.logits, labels, weights)
            loss    = loss / args.grad_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * args.grad_accumulation_steps

            if step % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "train_loss": f"{epoch_loss / step:.4f}",
                "lr":         f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_train_loss = epoch_loss / len(train_loader)
        LOGGER.info("Epoch %d | train loss: %.4f", epoch, avg_train_loss)

        # ---- Val loss ----
        val_loss = run_val_loss(model, val_loader, device)
        LOGGER.info("Epoch %d | val loss:   %.4f", epoch, val_loss)

        # ---- Val accuracy ----
        LOGGER.info("Epoch %d | running val accuracy...", epoch)
        run_val_accuracy(
            model, tokenizer, val_records, device,
            max_new_tokens=args.val_max_new_tokens,
            samples_per_lang=args.val_acc_samples_per_lang,
            gen_batch_size=args.val_gen_batch_size,
            seed=args.seed,
        )

        # ---- Checkpoint + early stopping ----
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_path         = output_dir / "best"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            LOGGER.info("Epoch %d | saved best checkpoint (val_loss=%.4f)", epoch, val_loss)
        else:
            epochs_no_improve += 1
            LOGGER.info(
                "Epoch %d | no improvement for %d/%d epochs",
                epoch, epochs_no_improve, args.early_stopping_patience,
            )
            if epochs_no_improve >= args.early_stopping_patience:
                LOGGER.info("Early stopping triggered after epoch %d.", epoch)
                break

        model.train()

    # ---- Save final ----
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    LOGGER.info("Saved final checkpoint to %s", final_path)
    LOGGER.info("Best val loss: %.4f", best_val_loss)

# Logger + args
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
    parser.add_argument("--train_data",   default="outputs/train.jsonl")
    parser.add_argument("--output_dir",   required=True)

    # Training
    parser.add_argument("--batch_size",              type=int,   default=4)
    parser.add_argument("--grad_accumulation_steps", type=int,   default=8)
    parser.add_argument("--epochs",                  type=int,   default=10)
    parser.add_argument("--lr",                      type=float, default=2e-4)
    parser.add_argument("--max_length",              type=int,   default=1536)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Stop after N epochs with no val loss improvement")

    # Validation
    parser.add_argument("--val_fraction",             type=float, default=0.1)
    parser.add_argument("--val_max_new_tokens",       type=int,   default=512)
    parser.add_argument("--val_acc_samples_per_lang", type=int,   default=VAL_ACC_SAMPLES_PER_LANG)
    parser.add_argument("--val_gen_batch_size",       type=int,   default=8)

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Misc
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--log_level",     default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file_name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level, args.log_file_name)
    train(args)


if __name__ == "__main__":
    main()