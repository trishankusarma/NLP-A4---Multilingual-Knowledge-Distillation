from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_vllm_llm, prompt_vllm

LOGGER = logging.getLogger(__name__)

ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)

LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "english": "ENGLISH",
    "hindi":   "HINDI",
    "bengali": "BENGALI",
    "kannada": "KANNADA",
    "tamil":   "TAMIL",
}

LANGUAGE_INSTRUCTIONS = {
    "en":      "Reason step by step in English and respond",
    "hindi":   "Reason step by step in English and respond",
    "bengali": "Reason step by step in English and respond",
    "kannada": "Reason step by step in English and respond",
    "tamil":   "Reason step by step in English and respond",
}

# Helpers
def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _canonical_language(lang: str) -> str:
    aliases = {
        "en": "en", "eng": "en", "english": "en",
        "hi": "hindi", "hindi": "hindi",
        "bn": "bengali", "bengali": "bengali",
        "kn": "kannada", "kannada": "kannada",
        "ta": "tamil", "tamil": "tamil",
    }
    return aliases.get(str(lang).strip().lower(), "en")


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {opt}" for idx, opt in enumerate(options)
    )


def _build_instruction(row: dict) -> str:
    options = row.get("options", [])
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def build_prompt(instruction: str, language: str, tokenizer) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, "Reason step by step in English and respond")
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
        {"role": "user",   "content": instruction},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"System: {system_content}\nUser: {instruction}\nAssistant:"


def parse_answer(raw: str, n_options: int) -> str:
    valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]

    # Primary: #### ANSWER: X
    match = ANSWER_TAG_RE.search(raw)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter

    # Fallback: last standalone A-J letter in valid range
    all_letters = LAST_LINE_LETTER_RE.findall(raw)
    for letter in reversed(all_letters):
        if letter.upper() in valid_letters:
            return letter.upper()

    return ""


def load_test_data(test_data: str) -> list[dict]:
    rows = []
    with open(test_data, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    LOGGER.info("Loaded %d test rows from %s", len(rows), test_data)
    return rows


def merge_adapter(base_model: str, adapter_path: str, tmp_dir: str, dtype: str) -> str:
    """Merge LoRA adapter into base model and save for vLLM loading."""
    LOGGER.info("Merging LoRA adapter from %s into %s ...", adapter_path, base_model)
    torch_dtype = torch.float16 if dtype == "float16" else \
                  torch.bfloat16 if dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    LOGGER.info("Merged model saved to %s", tmp_dir)
    return tmp_dir

# Args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Base student model path or HuggingFace ID")
    parser.add_argument("--adapter_path", default="",
                        help="Optional LoRA adapter path (output_dir/best or final)")
    parser.add_argument("--test_data", required=True,
                        help="Test JSONL path")
    parser.add_argument("--output_predictions", required=True,
                        help="Predictions JSONL output path")
    parser.add_argument("--report_file", required=True,
                        help="Metrics report text file path")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype — float16 for V100, bfloat16 for A100")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max sequence length for vLLM")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()

# Main
def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    # 1. Load model — merge LoRA if adapter provided
    if args.adapter_path:
        merged_dir = str(Path(args.output_predictions).parent / "merged_model")
        model_path = merge_adapter(
            args.base_model, args.adapter_path, merged_dir, args.dtype,
        )
    else:
        model_path = args.base_model
        LOGGER.info("No adapter path provided — running base model directly")

    llm, tokenizer = load_vllm_llm(
        model_id=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    # 2. Load test data and build prompts
    rows        = load_test_data(args.test_data)
    instructions = [_build_instruction(r) for r in rows]
    languages    = [_canonical_language(r.get("language", "en")) for r in rows]
    prompts      = [
        build_prompt(inst, lang, tokenizer)
        for inst, lang in zip(instructions, languages)
    ]

    # 3. Generate
    LOGGER.info("Running inference on %d samples...", len(prompts))
    generations = prompt_vllm(
        llm,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        repetition_penalty=1.1,
        use_tqdm=True,
    )

    # 4. Parse answers
    predictions = []
    for row, instruction, lang, generation in zip(rows, instructions, languages, generations):
        gold      = str(row.get("answer", "")).upper()[:1]
        n_options = len(row.get("options", []))
        predicted = parse_answer(generation, n_options)

        predictions.append({
            "language":         lang,
            "subject":          row.get("subject", ""),
            "question":         row.get("question", instruction),
            "gold_answer":      gold,
            "predicted_answer": predicted,
            "generation":       generation,
        })

    # 5. Write predictions JSONL
    pred_path = Path(args.output_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as fp:
        for pred in predictions:
            fp.write(json.dumps(pred, ensure_ascii=False) + "\n")
    LOGGER.info("Saved predictions to %s", pred_path)

    # 6. Compute accuracy per language + overall 
    lang_correct: dict[str, int] = defaultdict(int)
    lang_total:   dict[str, int] = defaultdict(int)

    for pred in predictions:
        lang = pred["language"]
        lang_total[lang]   += 1
        lang_correct[lang] += int(pred["predicted_answer"] == pred["gold_answer"])

    overall_correct = sum(lang_correct.values())
    overall_total   = sum(lang_total.values())

    # 7. Write metrics report
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Map canonical codes back to display names
    display_map = {
        "en":      "ENGLISH",
        "hindi":   "HINDI",
        "bengali": "BENGALI",
        "kannada": "KANNADA",
        "tamil":   "TAMIL",
    }

    for lang_code, display in display_map.items():
        if lang_total[lang_code] == 0:
            continue
        acc  = 100.0 * lang_correct[lang_code] / lang_total[lang_code]
        line = f"{display} ACCURACY: {acc:.2f}"
        lines.append(line)
        LOGGER.info(line)

    overall_acc  = 100.0 * overall_correct / max(overall_total, 1)
    overall_line = f"OVERALL ACCURACY: {overall_acc:.2f}"
    lines.append(overall_line)
    LOGGER.info(overall_line)

    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    LOGGER.info("Saved metrics report to %s", report_path)


if __name__ == "__main__":
    main()