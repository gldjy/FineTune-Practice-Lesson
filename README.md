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

## 脚本用法

三个脚本都在 `src/` 下，用 `poetry run python src/<脚本名>.py` 运行。

### 准备数据 `src/prepare_data.py`

读取 `datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv` 里的法律选择题，清洗题面后打乱切分，输出 `law_train.jsonl`、`law_val.jsonl`、`law_test.jsonl`：

```bash
# 默认行为：输出到 datasets/，按 17:1:2 切分全部样本
poetry run python src/prepare_data.py

# 换输出目录、切分比例和采样量（打乱后取前 1000 条，按 8:1:1 切）
poetry run python src/prepare_data.py --out-dir datasets/mini --max-samples 1000 --split-ratio 8,1,1
```

### 训练 `src/train_qwen3.py`

LoRA 微调，默认训练 1 轮，batch-size 2 × grad-accum 8。适配器存到 `output/law-qwen3/final`，日志写到 `logs/`：

```bash
# 从头训练
poetry run python src/train_qwen3.py

# 从存档点继续训练
poetry run python src/train_qwen3.py --resume-ckpt output/law-qwen3/checkpoint-1000

# 换数据、换轮数、换学习率
poetry run python src/train_qwen3.py \
  --train-file datasets/mini/law_train.jsonl \
  --val-file datasets/mini/law_val.jsonl \
  --epoch 2 --lr 1e-4
```

显存不够时优先调小 `--batch-size` 并按倍数调大 `--grad-accum`，保持两者乘积不变，学习率就不用跟着改。

### 测试 `src/test_qwen3.py`

先评指定的模型，结果文件按运行时间命名写到 `logs/test_results/`：

```bash
# 默认：挂 output/law-qwen3/final 适配器，评完整测试集
poetry run python src/test_qwen3.py

# 只评原始模型（不挂适配器）做基准
poetry run python src/test_qwen3.py --ckpt none

# 换别的存档点或换测试集
poetry run python src/test_qwen3.py \
  --ckpt output/law-qwen3/checkpoint-1000 \
  --test-file datasets/mini/law_test.jsonl
```

```python
import pandas as pd

d = pd.read_json("logs/test_results/test-xxxx.jsonl", lines=True)
print(d.correct.mean())
```

全量测试会比较久，想先小规模验证就用上面 `datasets/mini` 那种小测试集。
