"""Evaluation of Qwen3 (base model or a LoRA checkpoint) on the law multiple-choice test set."""

import argparse
import json
import os
import random
import re
from datetime import datetime

import torch
from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM

DEFAULT_SEED = 42

# Fixed generation setting
MAX_NEW_TOKENS = 64

ANSWER_PATTERN = re.compile(r"\b([A-D])\b")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 (base model or a LoRA checkpoint) on the law multiple-choice test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default="./qwen3", help="Path to the base model directory"
    )
    parser.add_argument(
        "--ckpt",
        default="output/law-qwen3/final",
        help="LoRA checkpoint directory to evaluate; pass 'none' to evaluate the base model",
    )
    parser.add_argument(
        "--test-file",
        default="datasets/law_test.jsonl",
        help="Path to the test JSONL file",
    )
    parser.add_argument(
        "--result-dir",
        default="logs",
        help="Directory for the result JSONL files",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def load_test_data(path):
    """Loads the non-empty JSONL records of the test set."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_letter(text):
    """Extracts the A/B/C/D answer letter from the model output."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1) if match else None


def generate_answer(model, tokenizer, instruction, input_text, max_new_tokens):
    """Generates an answer for a single sample with greedy decoding."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_test(model, tokenizer, test_data, out_file, meta, max_new_tokens):
    """Runs one evaluation pass; out_file gets one meta line plus one JSONL record per sample."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    model.eval()
    correct = 0
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for i, sample in enumerate(test_data):
            expected = sample["output"].strip()[0].upper()  # Correct letter, e.g. "A"
            pred = generate_answer(
                model, tokenizer, sample["instruction"], sample["input"], max_new_tokens
            )
            pred_letter = extract_letter(pred)
            ok = pred_letter == expected
            correct += int(ok)
            # Full question, options and model output, no truncation.
            record = {
                "index": i + 1,
                "input": sample["input"],
                "expected": expected,
                "predicted": pred_letter,
                "correct": ok,
                "output": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{i + 1:3d}/{len(test_data)}] expected={expected}  "
                f"predicted={pred_letter}  {'right' if ok else 'wrong'}"
            )

    return correct


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_data = load_test_data(args.test_file)
    if not test_data:
        raise SystemExit(f"No test samples found in {args.test_file}")
    print(f"Test samples: {len(test_data)}")

    print(f"Loading tokenizer and model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto"
    )

    if args.ckpt.lower() in ("none", "-"):
        ckpt = None  # No adapter: evaluate the base model.
        label = "base model"
        print("Evaluating the base model (no LoRA checkpoint) ...")
    else:
        ckpt = args.ckpt
        label = "LoRA"
        print(f"Loading LoRA checkpoint from {ckpt} ...")
        model = PeftModel.from_pretrained(model, ckpt)
        print("Evaluating the LoRA-tuned model ...")

    now = datetime.now()  # noqa: DTZ005
    out_file = os.path.join(args.result_dir, f"test-{now:%Y%m%d-%H%M%S}.jsonl")
    meta = {
        "timestamp": now.isoformat(timespec="seconds"),
        "ckpt": ckpt,
        "model_dir": args.model_dir,
        "test_file": args.test_file,
        "seed": args.seed,
        "samples": len(test_data),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    correct = run_test(model, tokenizer, test_data, out_file, meta, MAX_NEW_TOKENS)

    print(f"\n{label} accuracy: {correct}/{len(test_data)} = {correct / len(test_data):.2%}")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
