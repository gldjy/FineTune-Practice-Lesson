# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 法律选择题 LoRA 微调（基于 DISC-Law-SFT 的 exam 类型）。

和医学项目 train_qwen3.py 几乎一样，只改了数据路径和输出路径，
并且训练过程会记录到日志文件 law_data/training_log.txt。

用法：先 python prepare_data_law.py，再 python train_qwen3_law.py
"""
import glob
import json
import os
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

# ================= 可配置参数 =================
MODEL_ID = "Qwen/Qwen3-0.6B"
TRAIN_FILE = "datasets/law_train.jsonl"        # 训练数据
DEV_FILE = "datasets/law_val.jsonl"            # 验证集
OUTPUT_DIR = "./output/law-qwen3"              # 输出到 output 下的新文件夹
LOG_FILE = "datasets/DISC-Law-SFT/training_log.txt"         # 训练日志文件
MAX_LENGTH = 512

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

LR = 2e-4
EPOCHS = 2      # 医学项目经验：2 轮最佳，3 轮会过拟合
BATCH_SIZE = 4
GRAD_ACCUM = 4
# =============================================


# 训练日志回调：把每个 log（loss / eval_loss 等）写到文件
class LogToFileCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        # 只有文件不存在或为空时才写表头（续训时不覆盖之前的日志）
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("=== 法律选择题 LoRA 微调训练日志 ===\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False, default=str) + "\n")


# 1. 下载并加载模型
print("正在下载模型 ...")
model_dir = snapshot_download(MODEL_ID, cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", dtype=torch.bfloat16)
model.enable_input_require_grads()


# 2. 数据预处理
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
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


train_df = pd.read_json(TRAIN_FILE, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
print(f"训练数据：{len(train_dataset)} 条")

dev_df = pd.read_json(DEV_FILE, lines=True)
dev_ds = Dataset.from_pandas(dev_df)
eval_dataset = dev_ds.map(process_func, remove_columns=dev_ds.column_names)
print(f"验证数据：{len(eval_dataset)} 条")


# 3. LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


# 4. 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=100,          # 每 100 步存一次档（更频繁，重启损失更小）
    eval_strategy="steps",
    eval_steps=200,
    gradient_checkpointing=True,
    bf16=True,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[LogToFileCallback(LOG_FILE)],
)

print("开始训练 ...")
# 自动判断：有存档点就接着练，没有就从头开始
checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
if checkpoints:
    print(f"发现存档点，从最近的继续训练 ...")
    result = trainer.train(resume_from_checkpoint=True)
else:
    print("无存档点，从头开始训练 ...")
    result = trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")

# 写训练总结
summary = {
    "训练完成": True,
    "总步数": result.global_step,
    "总耗时_秒": round(result.metrics.get("train_runtime", 0), 1),
    "最终训练loss": result.metrics.get("train_loss"),
}
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write("\n=== 训练总结 ===\n")
    f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

print(f"训练完成，模型已保存到 {OUTPUT_DIR}/final")
print(f"训练日志已保存到 {LOG_FILE}")
