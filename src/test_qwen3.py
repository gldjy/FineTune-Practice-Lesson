import json
import re

import torch
from modelscope import AutoTokenizer, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER = "output/law-qwen3/final"  # 训练好的 LoRA 适配器
TEST_FILE = "datasets/law_test.jsonl"  # 法律测试集（未参与训练）
NUM_TEST = 1000  # 测前 1000 条

base_dir = snapshot_download(MODEL_ID, cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained(
    base_dir, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_dir, dtype=torch.bfloat16, device_map="auto"
)


def generate_answer(model, instruction, input_text):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_letter(text):
    """从模型回答里提取 A/B/C/D 字母"""
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


test_data = []
with open(TEST_FILE, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= NUM_TEST:
            break
        test_data.append(json.loads(line))


def run_test(model, out_file, label):
    model.eval()
    correct = 0
    result_lines = [f"===== {label} ====="]
    for i, d in enumerate(test_data):
        expected = d["output"][0]  # 正确答案字母，如 "A"
        pred = generate_answer(model, d["instruction"], d["input"])
        pred_letter = extract_letter(pred)
        ok = pred_letter == expected
        if ok:
            correct += 1
        mark = "对" if ok else "错"
        line = (
            f"[{i+1:3d}/{len(test_data)}] 正确答案={expected}  模型答={pred_letter}  "
            f"{mark}  | 原始输出: {pred[:30]}"
        )
        print(line)
        result_lines.append(line)
        result_lines.append(f"    题目: {d['input'][:100]}")
        result_lines.append("")

    summary = f"===== {label} 准确率: {correct}/{len(test_data)} = {correct/len(test_data)*100:.1f}% ====="
    print(f"\n{summary}\n")
    result_lines.append(summary)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"结果已保存到 {out_file}\n")


print("【第 1 步】测试微调前的原始模型 ...")
run_test(model, "test_1000_law/law_test_result_baseline.txt", "微调前(原始模型)")

model = PeftModel.from_pretrained(model, ADAPTER)
print("【第 2 步】测试微调后的模型 ...")
run_test(model, "test_1000_law/law_test_result.txt", "微调后(LoRA)")

print("全部完成！请对比 law_test_result_baseline.txt 和 law_test_result.txt 两个文件。")
