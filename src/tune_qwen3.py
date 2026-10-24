"""One LoRA hyperparameter trial: trains, measures, records metrics.json.

Unlike ``train_qwen3.py`` (fixed LoRA shape, eval on the val loss only), this script
exists for hyperparameter search: every knob that matters for a sweep is a flag, no
checkpoints are kept during training, and the run directory carries a machine-readable
``metrics.json`` plus the trainer loss log, so a sweep can be compared afterwards.

Example:
    poetry run python src/tune_qwen3.py --run-name mini-lr2e-4 \\
        --dataset-dir datasets/mini --lr 2e-4 --epoch 1
"""

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)  # 防止 Cuda OOM

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from dataset_splits import find_split_file
from datasets import Dataset
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

DEFAULT_SEED = 42

# 和 train_qwen3.py 保持一致，保证「调参结果」能直接搬回正式训练
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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Run one LoRA hyperparameter trial and record its metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="Run id, e.g. mini-lr2e-4")
    parser.add_argument(
        "--output-dir", default="output/opt", help="Root directory for all runs"
    )
    parser.add_argument(
        "--dataset-dir", default="datasets/mini", help="Dataset variant directory"
    )
    parser.add_argument(
        "--model-dir", default="./qwen3", help="Path to the base model directory"
    )

    # 训练超参数；--lr 默认与 train_qwen3.py 一致，取调参得到的 3e-4
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epoch", type=float, default=1.0, help="Training epochs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap on optimizer steps; 0 means no cap",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.0, help="Warmup ratio of total steps"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--scheduler",
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="LR scheduler type",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Grad clip")

    # LoRA 形状
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout"
    )

    parser.add_argument("--max-length", type=int, default=1024, help="Token cap")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--eval-batch-size", type=int, default=4, help="Batch size for proxy accuracy"
    )
    parser.add_argument("--note", default="", help="Free-form note stored in metrics")
    return parser.parse_args()


class LogToFileCallback(TrainerCallback):
    """把 Trainer 每条日志追加写进文件，方便之后画曲线"""

    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False, default=str) + "\n")


def build_process_func(tokenizer, max_length):
    """把样本转成 chat 格式的 input_ids/labels，并把 prompt 部分 mask 掉"""

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


def load_records(path):
    """读取 JSONL 原始记录（保留 output 字符串，用来区分单选/多选）"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(path, tokenizer, max_length):
    """读取 JSONL 并转成训练特征"""
    df = pd.read_json(path, lines=True)
    ds = Dataset.from_pandas(df)
    return ds.map(
        build_process_func(tokenizer, max_length), remove_columns=ds.column_names
    )


@torch.no_grad()
def answer_stats(model, dataset, collator, batch_size, records):
    """Teacher-forced 代理准确率。

    答案只有 1~4 个字母，逐位取 argmax 与标签比对：整个答案区全对算 exact match，
    同时给出 token 级准确率。比逐样本 generate 快一个数量级，适合每轮都算。
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    exact = total = tok_ok = tok_total = 0
    per_kind = {"single": [0, 0], "multiple": [0, 0]}
    for step, batch in enumerate(loader):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        pred = logits[:, :-1].argmax(-1)
        target = batch["labels"][:, 1:]
        mask = target != -100
        hit = (pred == target) & mask
        tok_ok += int(hit.sum())
        tok_total += int(mask.sum())
        ok = ((pred == target) | ~mask).all(dim=1).tolist()
        for offset, sample_ok in enumerate(ok):
            index = step * batch_size + offset
            if index >= len(records):
                break
            kind = "multiple" if len(records[index]["output"]) > 1 else "single"
            per_kind[kind][0] += int(sample_ok)
            per_kind[kind][1] += 1
            exact += int(sample_ok)
            total += 1

    def rate(pair):
        return pair[0] / pair[1] if pair[1] else 0.0

    return {
        "exact_match": rate([exact, total]),
        "token_acc": tok_ok / tok_total if tok_total else 0.0,
        "single_acc": rate(per_kind["single"]),
        "single_n": per_kind["single"][1],
        "multiple_acc": rate(per_kind["multiple"]),
        "multiple_n": per_kind["multiple"][1],
        "n": total,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.output_dir) / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.jsonl"
    log_file.write_text("", encoding="utf-8")

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

    train_dataset = load_dataset(train_file, tokenizer, args.max_length)
    val_dataset = load_dataset(val_file, tokenizer, args.max_length)
    val_records = load_records(val_file)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        ),
    )
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # transformers 5 只认 warmup_steps，这里按 ratio 换算成步数，方便和常见配方对齐
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch))
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else int(math.ceil(steps_per_epoch * args.epoch))
    )
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    print(
        f"Schedule: steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup_steps={warmup_steps}"
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        lr_scheduler_type=args.scheduler,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[LogToFileCallback(log_file)],
    )

    print("Training ...")
    result = trainer.train()
    runtime = result.metrics.get("train_runtime", 0.0)

    # 训练时的激活缓存会一直占着显存，不释放的话下面的整词表 logits（每批
    # ~1.5 GB）会顶到 8 GB 上限，触发 Windows 的共享内存换页，几十秒都算不完一批
    torch.cuda.empty_cache()
    print(
        f"GPU memory after training: "
        f"{torch.cuda.memory_allocated() / 2**20:.0f} MB allocated, "
        f"{torch.cuda.memory_reserved() / 2**20:.0f} MB reserved"
    )

    print("Evaluating ...")
    eval_metrics = trainer.evaluate(
        eval_dataset=val_dataset, metric_key_prefix="eval"
    )
    stats = answer_stats(
        trainer.model, val_dataset, collator, args.eval_batch_size, val_records
    )

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    metrics = {
        "run": args.run_name,
        "note": args.note,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_dir": args.dataset_dir,
        "train_file": train_file,
        "val_file": val_file,
        "config": {
            "lr": args.lr,
            "epoch": args.epoch,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "max_grad_norm": args.max_grad_norm,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "train": {
            "steps": result.global_step,
            "runtime_seconds": round(runtime, 1),
            "train_loss": result.metrics.get("train_loss"),
            "n_samples": len(train_dataset),
            "samples_per_second": round(
                result.metrics.get("train_samples_per_second", 0.0), 2
            ),
        },
        "eval": {"eval_loss": eval_metrics.get("eval_loss"), **stats},
        "gpu": {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else 0,
            "allocated_before_train_mb": round(gpu_before / 2**20, 1),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 汇总行，画图和横向对比都只读这一个文件
    summary_path = Path(args.output_dir) / "runs.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run": args.run_name,
                    "dataset": Path(args.dataset_dir).name,
                    "lr": args.lr,
                    "epoch": args.epoch,
                    "warmup_ratio": args.warmup_ratio,
                    "scheduler": args.scheduler,
                    "weight_decay": args.weight_decay,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "steps": result.global_step,
                    "train_loss": metrics["train"]["train_loss"],
                    "eval_loss": metrics["eval"]["eval_loss"],
                    "exact_match": stats["exact_match"],
                    "single_acc": stats["single_acc"],
                    "multiple_acc": stats["multiple_acc"],
                    "runtime_seconds": metrics["train"]["runtime_seconds"],
                    "peak_allocated_mb": metrics["gpu"]["peak_allocated_mb"],
                    "timestamp": metrics["timestamp"],
                    "note": args.note,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        f"DONE {args.run_name}: eval_loss={metrics['eval']['eval_loss']:.4f} "
        f"exact={stats['exact_match']:.2%} "
        f"single={stats['single_acc']:.2%} multiple={stats['multiple_acc']:.2%} "
        f"steps={result.global_step} runtime={runtime:.0f}s"
    )
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
