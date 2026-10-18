# FineTune-Practice-Lesson

我上晚十。

## 快速开始

1. 安装 [Poetry](https://python-poetry.org/docs/#installation) 依赖管理工具。

2. 安装 [Python](https://www.python.org/downloads/) 3.12 或以上版本。

3. 克隆仓库到本地。

4. 安装 Python 依赖（这将包含 Cuda 12.6 版本的 PyTorch）：
   ```bash
   poetry install
   ```

5. 下载 Qwen3-0.6B 模型文件到 `qwen3/` 目录下：
   ```bash
   poetry run modelscope download --model Qwen/Qwen3-0.6B --local_dir ./qwen3
   ```

6. 下载 DISC-Law-SFT 法律领域数据集到 `datasets/` 的对应目录下：
   ```bash
   poetry run modelscope download --dataset AI-ModelScope/DISC-Law-SFT --local_dir ./datasets/DISC-Law-SFT
   ```

7. 生成两个数据集变体，后面的脚本都用 `--dataset-dir` 指向其中一个（见[脚本用法](#脚本用法)）：
   ```bash
   # full：21009 条、17:1:2，正经训练和评测用
   poetry run python src/prepare_data.py

   # mini：1000 条、8:1:1，几十秒跑通流程用
   poetry run python src/prepare_data.py --dataset-dir datasets/mini --max-samples 1000 --split-ratio 8,1,1
   ```

## 脚本用法

三个脚本都在 `src/` 下，用 `poetry run python src/<脚本名>.py` 运行。

数据集按目录分成两个变体，三个脚本都用 `--dataset-dir` 指向其中一个。目录里的 `*train*.jsonl` / `*val*.jsonl`（也认 `*dev*`）/ `*test*.jsonl` 由脚本自动识别，不用写具体文件名：

| 变体 | 目录（默认） | 生成方式 | 规模与用途 |
| --- | --- | --- | --- |
| full | `datasets/full/` | `prepare_data.py` 默认参数 | 21009 条、17:1:2，正经训练和评测 |
| mini | `datasets/mini/` | 加 `--max-samples 1000 --split-ratio 8,1,1` | 1000 条、8:1:1，几十秒跑通整个流程 |

### 准备数据 `src/prepare_data.py`

读取 `datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv` 里的法律选择题，清洗题面后打乱切分，输出 `law_train.jsonl`、`law_val.jsonl`、`law_test.jsonl`：

```bash
# full 变体：输出到 datasets/full，丢掉长样本，其余按 17:1:2 切分
poetry run python src/prepare_data.py

# mini 变体：打乱后取前 1000 条，按 8:1:1 切
poetry run python src/prepare_data.py --dataset-dir datasets/mini --max-samples 1000 --split-ratio 8,1,1

# 更换丢弃长样本的长度阈值
poetry run python src/prepare_data.py --max-length 1280
```

源数据里约一半是多项选择题（答案区会列出多个正确选项），脚本按答案区行数区分题型，单选和多选各用一套 instruction，多选题的答案原样保留成多行。

### 训练 `src/train_qwen3.py`

LoRA 微调，默认训练 1 轮，batch-size 2 × grad-accum 8。适配器存到 `output/law-qwen3/final`，日志写到 `logs/`：

```bash
# 从头训练（full 变体，默认）
poetry run python src/train_qwen3.py

# 从存档点继续训练
poetry run python src/train_qwen3.py --resume-ckpt output/law-qwen3/checkpoint-1000

# 用 mini 变体先跑通流程，适配器写到单独目录，免得覆盖 full 的结果
poetry run python src/train_qwen3.py \
  --dataset-dir datasets/mini \
  --output-dir output/law-qwen3-mini

# 换轮数、换学习率
poetry run python src/train_qwen3.py --epoch 2 --lr 1e-4
```

显存不够时优先调小 `--batch-size` 并按倍数调大 `--grad-accum`，保持两者乘积不变，学习率就不用跟着改。

### 测试 `src/test_qwen3.py`

先评指定的模型，结果文件按运行时间命名写到 `--result-dir`（默认 `logs/`）下：

```bash
# 默认：挂 output/law-qwen3/final 适配器，评 full 的测试集
poetry run python src/test_qwen3.py

# 只评原始模型（不挂适配器）做基准
poetry run python src/test_qwen3.py --ckpt none

# 用 mini 变体和 mini 适配器快速验证
poetry run python src/test_qwen3.py \
  --dataset-dir datasets/mini \
  --ckpt output/law-qwen3-mini/final

# 换别的存档点
poetry run python src/test_qwen3.py --ckpt output/law-qwen3/checkpoint-1000
```

结果文件里 `expected` 和 `predicted` 都是选中的选项字母集合（多选形如 `AD`），两者完全一致才算对；控制台会另按单选、多选分别给出准确率。

```python
import pandas as pd

d = pd.read_json("logs/test-xxxx.jsonl", lines=True)
print(d.correct.mean())
```

full 的测试集跑一遍会比较久；想先小规模验证就加 `--dataset-dir datasets/mini`。
