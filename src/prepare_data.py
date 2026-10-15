import json
import random

import pandas as pd

# ---- 原始数据路径 ----
RAW_CSV = "datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv"

# ---- 输出路径（放在新建的数据文件夹 law_data 下）----
OUT_TRAIN = "datasets/law_train.jsonl"
OUT_VAL = "datasets/law_val.jsonl"
OUT_TEST = "datasets/law_test.jsonl"

# ---- 切分数量（可用一小部分，改这里）----
TRAIN_SIZE = 15000  # 训练集（总共 21054 条选择题，取 1.5 万）
VAL_SIZE = 500  # 验证集
TEST_SIZE = 1000  # 测试集

# ---- 随机种子（保证每次切分结果一样）----
SEED = 42

SYSTEM_PROMPT = (
    "你是一名法律考试助手。请阅读下面的法律单项选择题，"
    "从 A、B、C、D 四个选项中选出唯一正确的答案。"
    "只输出正确选项的字母和内容，不要解释。"
)


def parse_input(text):
    """去掉第一行指令，只留「题目 + 选项」部分"""
    if "\n" in text:
        text = text.split("\n", 1)[1]  # 第一行是「请给出...推理过程：」，去掉
    text = text.replace("问题：", "题目：", 1)  # 统一叫「题目」
    return text.strip()


def parse_output(text):
    """答案 = 第一行（字母 + 答案内容），去掉后面的推理过程"""
    return text.split("\n")[0].strip()


def main():
    print("读取 CSV ...")
    df = pd.read_csv(RAW_CSV, encoding="utf-8", on_bad_lines="skip")
    # 只取 exam 类型（法律选择题）
    exam = df[df["id"].astype(str).str.startswith("exam", na=False)]
    print(f"法律选择题（exam 类型）共 {len(exam)} 条")

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
    random.seed(SEED)
    random.shuffle(items)

    # 切分 train / val / test
    train = items[:TRAIN_SIZE]
    val = items[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test = items[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    for path, data in [(OUT_TRAIN, train), (OUT_VAL, val), (OUT_TEST, test)]:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)
        print(f"已生成 {path}：{len(data)} 条")


if __name__ == "__main__":
    main()
