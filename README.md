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

7. 生成三个数据集变体，后面的脚本都用 `--dataset-dir` 指向其中一个（见[脚本用法](#脚本用法)）：
   ```bash
   # full：21006 条、17:1:2，正经训练和评测用
   poetry run python src/prepare_data.py

   # medium：6000 条、17:1:2，中等规模对照实验用
   poetry run python src/prepare_data.py --dataset-dir datasets/medium --max-samples 6000

   # mini：2000 条、17:1:2，几十秒跑通整个流程用
   poetry run python src/prepare_data.py --dataset-dir datasets/mini --max-samples 2000
   ```

## 脚本用法

三个脚本都在 `src/` 下，用 `poetry run python src/<脚本名>.py` 运行。

数据集按目录分成三个变体，三个脚本都用 `--dataset-dir` 指向其中一个。目录里的 `*train*.jsonl` / `*val*.jsonl`（也认 `*dev*`）/ `*test*.jsonl` 由脚本自动识别，不用写具体文件名：

| 变体 | 目录 | 一键生成命令 | 规模（train / val / test） | 用途 |
| --- | --- | --- | --- | --- |
| full | `datasets/full/` | `poetry run python src/prepare_data.py` | 21006 = 17855 / 1050 / 2101 | 正经训练和评测 |
| medium | `datasets/medium/` | `poetry run python src/prepare_data.py --dataset-dir datasets/medium --max-samples 6000` | 6000 = 5100 / 300 / 600 | 中等规模对照实验 |
| mini | `datasets/mini/` | `poetry run python src/prepare_data.py --dataset-dir datasets/mini --max-samples 2000` | 2000 = 1700 / 100 / 200 | 几十秒跑通整个流程 |

### 准备数据 `src/prepare_data.py`

读取 `datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv` 里的法律选择题，按下面的流水线生成 `law_train.jsonl`、`law_val.jsonl`、`law_test.jsonl`：

1. 丢掉 chat 格式下超过 `--max-length`（默认 640）token 的长样本；
2. 去重：同一道题只留一条。指纹取"问题句尾部 110 字 + 全部选项"，所以对题号前缀和案例背景的截取长度都不敏感；同一指纹下答案不一致时保留正确选项更多的那条，`--no-dedup` 可以关掉这一步；
3. 打乱顺序，`--max-samples` 取前 N 条；
4. 按案例背景分组切分：同一个案例背景下的多道小问整组进同一个子集，避免同一案例同时出现在训练和测试里。

**输出只有字母**：`output` 是排序去重后的选项字母（单选形如 `C`，多选形如 `ABD`），不抄选项原文；单选和多选各用一套 instruction。这是出于评测效率和显存开销优化的考虑。

```bash
# 三个变体（都是 17:1:2，默认参数生成的即 full）
poetry run python src/prepare_data.py
poetry run python src/prepare_data.py --dataset-dir datasets/medium --max-samples 6000
poetry run python src/prepare_data.py --dataset-dir datasets/mini --max-samples 2000

# 换丢弃长样本的阈值、换切分比例、关掉去重
poetry run python src/prepare_data.py --max-length 1280
poetry run python src/prepare_data.py --split-ratio 8,1,1
poetry run python src/prepare_data.py --no-dedup
```

源数据里约一半是多项选择题，脚本按正确选项字母的个数区分题型，让模型只回字母。

三个变体取的是同一洗牌序的前 N 条，所以 `mini ⊂ medium ⊂ full`。但每个变体的切分是各自独立做的，一个变体的测试集有可能落在另一个变体的训练集里。所以训练和评测要用同一个变体，别拿 full 训练出的模型去评 mini 或 medium 的测试集。

### 训练 `src/train_qwen3.py`

LoRA 微调，默认训练 1 轮，batch-size 4 × grad-accum 4（有效 batch 16）。适配器存到 `output/law-qwen3/final`，日志写到 `logs/`：

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

在 8 GB NVIDIA 显卡上 batch 4 × grad-accum 4 是实测最佳权衡点，同样的有效 batch 16 下比 2 × 8 更快，但再往上到 8 × 2 会因为发生内存交换而更慢。

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
