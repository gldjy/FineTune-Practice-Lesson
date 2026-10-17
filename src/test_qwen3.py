"""Evaluation of Qwen3 and its LoRA adapter on the law multiple-choice test set."""

import argparse
import json
import os
import random
import re

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
        description="Evaluate Qwen3 (base model and LoRA adapter) on the law multiple-choice test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default="./qwen3", help="Path to the base model directory"
    )
    parser.add_argument(
        "--adapter",
        default="output/law-qwen3/final",
        help="Path to the trained LoRA adapter",
    )
    parser.add_argument(
        "--test-file",
        default="datasets/law_test.jsonl",
        help="Path to the test JSONL file",
    )
    parser.add_argument(
        "--result-dir",
        default="logs/test_results",
        help="Directory for the result text files",
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


def run_test(model, tokenizer, test_data, out_file, label, max_new_tokens):
    """Runs one evaluation pass and writes the per-sample results to out_file."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    model.eval()
    correct = 0
    result_lines = [f"===== {label} ====="]
    for i, sample in enumerate(test_data):
        expected = sample["output"].strip()[0].upper()  # Correct letter, e.g. "A"
        pred = generate_answer(
            model, tokenizer, sample["instruction"], sample["input"], max_new_tokens
        )
        pred_letter = extract_letter(pred)
        ok = pred_letter == expected
        correct += int(ok)
        line = (
            f"[{i + 1:3d}/{len(test_data)}] answer={expected}  predicted={pred_letter}  "
            f"{'right' if ok else 'wrong'} | output: {pred[:32].replace('\n', ' ')}"
        )
        print(line)
        result_lines.append(line)
        result_lines.append(f"    question: {sample['input'][:100]}")
        result_lines.append("")

    accuracy = correct / len(test_data)
    summary = (
        f"===== {label} accuracy: {correct}/{len(test_data)} = {accuracy:.2%} ====="
    )
    print(f"\n{summary}\n")
    result_lines.append(summary)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"Results saved to {out_file}\n")
    return accuracy


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

    baseline_file = os.path.join(args.result_dir, "law_test_result_baseline.txt")
    tuned_file = os.path.join(args.result_dir, "law_test_result.txt")

    print("[1/2] Evaluating the base model ...")
    baseline_acc = run_test(
        model, tokenizer, test_data, baseline_file, "base model", MAX_NEW_TOKENS
    )

    model = PeftModel.from_pretrained(model, args.adapter)
    print("[2/2] Evaluating the LoRA-tuned model ...")
    tuned_acc = run_test(
        model, tokenizer, test_data, tuned_file, "LoRA", MAX_NEW_TOKENS
    )

    print(f"Base model accuracy: {baseline_acc:.1%}")
    print(f"LoRA accuracy: {tuned_acc:.1%}")
    print(f"Improvement: {tuned_acc - baseline_acc:+.1%}")
    print(f"Results saved to {baseline_file} and {tuned_file}")


if __name__ == "__main__":
    main()
