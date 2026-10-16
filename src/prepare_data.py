import json
import random
import re

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


def main():
    print("读取 CSV ...")
    df = pd.read_csv(RAW_CSV, encoding="utf-8")
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
