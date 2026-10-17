"""Splits the DISC-Law-SFT exam items into train/val/test JSONL files."""

import argparse
import json
import os
import random
import re

import pandas as pd

# ---- 原始数据路径 ----
RAW_CSV = "datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv"

# Defaults for the command-line options
DEFAULT_SPLIT_RATIO = "17,1,2"  # train:val:test
DEFAULT_SEED = 42

SYSTEM_PROMPT = (
    "你是一名法律知识问答助手。请阅读下面的法律单项选择题，"
    "从 A、B、C、D 四个选项中选出唯一正确的答案。"
    "只输出正确选项的字母和内容，不要解释，不可多选。"
)


def parse_split_ratio(value):
    """Parses a "train,val,test" ratio string into a tuple of positive integers."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected exactly 3 comma-separated parts, got {value!r}"
        )
    try:
        ratio = tuple(int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"ratio parts must be integers, got {value!r}"
        ) from None
    if any(part <= 0 for part in ratio):
        raise argparse.ArgumentTypeError(f"ratio parts must be positive, got {value!r}")
    return ratio


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split the DISC-Law-SFT exam items into train/val/test JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", default="datasets", help="Directory for the generated JSONL files"
    )
    parser.add_argument(
        "--split-ratio",
        type=parse_split_ratio,
        default=DEFAULT_SPLIT_RATIO,
        help="Train/val/test ratio, e.g. 17,1,2",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of shuffled samples to use; -1 uses all",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


# 源数据把「请给出...答案和解析」这类指令混在题面里：有的在开头，有的在结尾，
# 而且不是每道题都有，所以只能按内容判断，不能按行号删
INSTRUCTION_KEYWORDS = ("单选", "多选", "选择题", "解析", "答案")


def is_instruction(line):
    """判断一行是不是源数据自带的指令（而不是题干或选项）"""
    line = line.strip()
    # 指令都是短句；选项行（A. / B. 开头）里的关键词不算
    if len(line) > 30 or re.match(r"^[A-D][\.、．]", line):
        return False
    return any(keyword in line for keyword in INSTRUCTION_KEYWORDS)


def parse_input(text):
    """去掉题面里混着的指令行，只留「题目 + 选项」部分"""
    text = "\n".join(line for line in text.split("\n") if not is_instruction(line))
    text = text.replace("问题：", "题目：", 1)  # 统一叫「题目」
    return text.strip()


def parse_output(text):
    """答案 = 第一行（字母 + 答案内容），去掉后面的推理过程"""
    return text.split("\n")[0].strip()


def split_items(items, ratio):
    """Splits shuffled items by ratio; the rounding remainder goes to the test split."""
    total = len(items)
    ratio_sum = sum(ratio)
    n_train = total * ratio[0] // ratio_sum
    n_val = total * ratio[1] // ratio_sum
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, encoding="utf-8")
    # 只取 exam 类型（法律选择题）
    exam = df[df["id"].astype(str).str.startswith("exam", na=False)]
    print(f"Exam items found: {len(exam)}")

    # 转成 instruction/input/output
    items = []
    for _, row in exam.iterrows():
        items.append(
            {
                "instruction": SYSTEM_PROMPT,
                "input": parse_input(row["input"]),
                "output": parse_output(row["output"]),
            }
        )

    # 打乱顺序
    random.shuffle(items)
    if args.max_samples >= 0:
        items = items[: args.max_samples]
        print(f"Using the first {len(items)} shuffled samples (--max-samples)")

    train, val, test = split_items(items, args.split_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    for split, data in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(args.out_dir, f"law_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)
        print(f"Wrote {len(data)} samples to {path}")


if __name__ == "__main__":
    main()
