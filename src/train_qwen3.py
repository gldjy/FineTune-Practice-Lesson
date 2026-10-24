"""LoRA fine-tuning of Qwen3 on the DISC-Law-SFT multiple-choice dataset."""

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)  # 防止 Cuda OOM

import argparse
import json
import random
from datetime import datetime

import pandas as pd
import torch
from dataset_splits import find_split_file
from datasets import Dataset
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DEFAULT_SEED = 42

# Fixed hyperparameters
MAX_LENGTH = 1024

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3 on the DISC-Law-SFT multiple-choice dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default="./qwen3", help="Path to the base model directory"
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/full",
        help="Directory holding the train/val/test JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/law-qwen3",
        help="Directory for checkpoints and the final adapter",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for the per-run training log file",
    )
    parser.add_argument(
        "--epoch", type=int, default=1, help="Number of training epochs"
    )
    # 默认值取自 docs/opt.md 的调参结论：medium 变体上 3e-4 的验证损失最低（0.5069），
    # 2e-4 次之；full 变体的步数是 medium 的 3.5 倍，需要时可退回 2e-4 对照
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device train and eval batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--resume-ckpt",
        default=None,
        help="Checkpoint directory to resume from; trains from scratch when omitted",
    )
    return parser.parse_args()


class LogToFileCallback(TrainerCallback):
    """Appends every Trainer log record (loss, eval_loss, ...) to a file."""

    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False, default=str) + "\n")


def build_process_func(tokenizer, max_length):
    """Builds a mapping function that formats samples as chat inputs and masks the prompt."""

    def process_func(example):
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return process_func


def load_dataset(path, tokenizer, max_length):
    """Loads a JSONL dataset and tokenizes it into training features."""
    df = pd.read_json(path, lines=True)
    ds = Dataset.from_pandas(df)
    return ds.map(
        build_process_func(tokenizer, max_length), remove_columns=ds.column_names
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Every run gets its own log file, named after the start time.
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"train-{datetime.now():%Y%m%d-%H%M%S}.log")  # noqa: DTZ005

    # 先定位数据文件再加载模型，找不到就早点退出
    train_file = find_split_file(args.dataset_dir, "train")
    val_file = find_split_file(args.dataset_dir, "val")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")

    print(f"Loading tokenizer and model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    train_dataset = load_dataset(train_file, tokenizer, MAX_LENGTH)
    print(f"Training samples: {len(train_dataset)}")
    eval_dataset = load_dataset(val_file, tokenizer, MAX_LENGTH)
    print(f"Validation samples: {len(eval_dataset)}")

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        save_total_limit=3,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[LogToFileCallback(log_file)],
    )

    if args.resume_ckpt:
        print(f"Resuming training from {args.resume_ckpt} ...")
        result = trainer.train(resume_from_checkpoint=args.resume_ckpt)
    else:
        print("Training from scratch ...")
        result = trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)

    summary = {
        "resumed_from": args.resume_ckpt,
        "epochs": args.epoch,
        "global_step": result.global_step,
        "train_runtime_seconds": round(result.metrics.get("train_runtime", 0), 1),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

    print(f"Training finished, model saved to {final_dir}")
    print(f"Training log saved to {log_file}")


if __name__ == "__main__":
    main()
