from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any
import torch

from datasets import Dataset, concatenate_datasets

from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)

LANGUAGE_INSTRUCTIONS = {
    "english": "Reason step by step in English and respond",
    "hindi":   "Reason step by step in English and respond",
    "bengali": "Reason step by step in English and respond",
    "kannada": "Reason step by step in English and respond",
    "tamil":   "Reason step by step in English and respond",
}

LANGUAGE_CODES = ["en", "hindi", "bengali", "kannada", "tamil"]

WRONG_FRACTIONS = {
    "en":      1.0,
    "hindi":   1.0,
    "bengali": 1.0,
    "kannada": 1.0,
    "tamil":   1.0,
}

def get_gpu_config(requested_util: float):
    """Auto-adjust settings based on available GPU."""
    if not torch.cuda.is_available():
        return requested_util, "auto", 64
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    name = props.name.lower()
    
    if "v100" in name or vram_gb < 20:
        # 16GB V100 — be conservative
        return min(requested_util, 0.80), "float16", 32
    elif vram_gb < 40:
        # 32GB V100 or similar
        return min(requested_util, 0.85), "float16", 64
    else:
        # A100/H100 — your current setup
        return requested_util, "bfloat16", 256

def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def sample_datasets(
        samples_per_language: list[int],
        split: str = "test",
        seed: int = 42,
) -> Dataset:
    """Fetch and sample the requested number of rows for each language."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if sum(samples_per_language) <= 0:
        raise ValueError("--num_samples must request at least one sample")

    splits = []
    for lang, count in zip(LANGUAGES, samples_per_language):
        if count == 0:
            continue
        ds = MMLUPro(language=lang, split=split).get_unified_dataset()
        available = len(ds)
        if available == 0:
            LOGGER.warning("No rows found for language=%s, skipping", lang)
            continue
        if count > available:
            LOGGER.warning(
                "Requested %d samples for %s but only %d available — using all",
                count, lang, available,
            )
            count = available
        ds = ds.shuffle(seed=seed).select(range(count))
        LOGGER.info("Sampled %d rows for language=%s", len(ds), lang)
        splits.append(ds)

    return concatenate_datasets(splits)


def format_teacher_prompt(instruction: str, language: str) -> list[dict]:
    """Build a language-aware chat message list for the teacher."""
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, "Respond in English.")
    system_content = (
        "You are an expert reasoning assistant. "
        f"{lang_instruction} "
        "Think step by step. "
        "Wrap ALL of your reasoning inside <reasoning>...</reasoning> tags. "
        "After the closing tag, end your response with exactly: #### ANSWER: [LETTER] "
        "where [LETTER] is one of A-J. Do not add anything after the answer line."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": instruction},
    ]


def generate_and_parse(
        llm,
        tokenizer,
        prompts: list[list[dict]],
        max_new_tokens: int = 1024,
) -> list[dict[str, str]]:
    """
    Query teacher model in a single vLLM call and extract reasoning + final answer.
    Returns one dict per prompt with keys: reasoning, final_answer, raw_generation.
    """
    raw_generations = prompt_vllm(
        llm,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        repetition_penalty=1.0,
        use_tqdm=True,
    )

    results = []
    for raw in raw_generations:
        # Extract <reasoning> block
        reasoning_match = REASONING_BLOCK_RE.search(raw)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract #### ANSWER: X — with fallback to last standalone letter
        answer_match = ANSWER_RE.search(raw)
        if answer_match:
            final_answer = answer_match.group(1).upper()
        else:
            all_letters = re.findall(r'\b([A-J])\b', raw, re.IGNORECASE)
            final_answer = all_letters[-1].upper() if all_letters else ""

        results.append({
            "reasoning":      reasoning,
            "final_answer":   final_answer,
            "raw_generation": raw,
        })

    return results


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )
    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--num_samples must contain only integers") from exc
    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    parser.add_argument("--teacher_model", required=True,
                        help="Hugging Face path to the teacher model")
    parser.add_argument("--num_samples", type=str, required=True,
                        help="Comma-separated sample counts for english,hindi,bengali,kannada,tamil")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL path for train corpus")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate per sample")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Target fraction of GPU memory for vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    # Auto-detect GPU and adjust settings
    gpu_util, dtype, max_num_seqs = get_gpu_config(args.gpu_memory_utilization)
    LOGGER.info("GPU config: util=%.2f, dtype=%s, max_num_seqs=%d", 
                gpu_util, dtype, max_num_seqs)

    samples_per_language = _parse_num_samples(args.num_samples)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sample dataset  
    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=args.seed,
    )
    rows = list(sampled)
    LOGGER.info("Collected %d samples total", len(rows))

    # 2. Load teacher
    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=gpu_util,   # using auto-detected value
        dtype=dtype,                       
        max_num_seqs=max_num_seqs,        
    )

    # 3. Build all prompts and submit in ONE vLLM call
    all_instructions = [_build_instruction(r) for r in rows]
    all_prompts = [
        format_teacher_prompt(inst, r["language"])
        for inst, r in zip(all_instructions, rows)
    ]

    LOGGER.info("Submitting all %d prompts to vLLM in one shot...", len(all_prompts))
    all_parsed = generate_and_parse(
        teacher, tokenizer, all_prompts,
        max_new_tokens=args.max_new_tokens,
    )
    LOGGER.info("Generation complete.")

    # 4. Build records
    all_records = []
    for row, instruction, parsed in zip(rows, all_instructions, all_parsed):
        gold = str(row.get("answer", "")).upper()[:1]

        # Validate answer is within option range
        n_options = len(row.get("options", []))
        valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]
        if parsed["final_answer"] not in valid_letters:
            parsed["final_answer"] = ""

        is_correct = parsed["final_answer"] == gold
        all_records.append({
            "question":           instruction,
            "reasoning":          parsed["reasoning"],
            "final_answer":       parsed["final_answer"],
            "gold_answer":        gold,
            "language":           row["language"],
            "subject":            row.get("subject"),
            "teacher_generation": parsed["raw_generation"],
            "correct":            is_correct,
        })

    # 5. Filter correct / sample incorrect
    correct   = [r for r in all_records if r["correct"]]
    incorrect = [r for r in all_records if not r["correct"]]

    LOGGER.info(
        "Teacher accuracy: %d / %d correct (%.1f%%)",
        len(correct), len(all_records),
        100 * len(correct) / max(len(all_records), 1),
    )

    random.seed(args.seed)

    # now sample incorrect samples as per language
    kept_incorrect = []
    for ln_code in LANGUAGE_CODES:
        
        incorrect_for_curr_language = [sample for sample in incorrect if sample["language"] == ln_code]
        n_wrong_to_keep = int(len(incorrect_for_curr_language) * WRONG_FRACTIONS[ln_code])
        kept_incorrect_lang = random.sample(incorrect_for_curr_language, min(n_wrong_to_keep, len(incorrect_for_curr_language)))
        kept_incorrect.extend(kept_incorrect_lang)

    LOGGER.info("Keeping %d incorrect samples total (language-specific fractions applied)",
            len(kept_incorrect))

    final_records = correct + kept_incorrect
    random.shuffle(final_records)
    LOGGER.info("Final dataset size: %d records", len(final_records))

    # 6. Write JSONL
    with output_path.open("w", encoding="utf-8") as fp:
        for record in final_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %d rows to %s", len(final_records), output_path)

    # Per-language breakdown
    lang_counts = Counter(r["language"] for r in final_records)
    for lang, cnt in sorted(lang_counts.items()):
        LOGGER.info("  %-10s : %d rows", lang, cnt)


if __name__ == "__main__":
    main()
