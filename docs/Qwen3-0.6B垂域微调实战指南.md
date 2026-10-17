# 领域大模型应用实践：Qwen3-0.6B 垂域微调实战

面向医学、法律、金融三个垂直领域，用 **Qwen3-0.6B** 做 LoRA 微调：下载模型 → 下载数据集 → 配置环境 → 数据处理成 `instruction/input/output` → 训练 → 测试对比微调前后效果。

> 本文档整理自课程 PPT 的技术部分与任务/评分要求，已去除学校、教师、班级等个人与机构标识，以及签到、实验室管理之类的行政事务。

## 目录

- [一、任务要求](#一任务要求)
- [二、评分规则](#二评分规则)
- [三、基准模型的选择](#三基准模型的选择)
- [四、数据集下载](#四数据集下载)
- [五、环境配置](#五环境配置)
- [六、数据处理](#六数据处理)
- [七、训练与测试](#七训练与测试)
- [八、算力预算与租用](#八算力预算与租用)
- [附：文件与命令速查](#附文件与命令速查)

---

## 一、任务要求

### 1.1 题目分配

每组一个题目，**题目按组号模 3 后给定**，对应医学、法律、金融三个领域。各题目的难易程度和数据量不尽相同。

### 1.2 组织形式

- 分小组进行，**每组 4 人**，选定组长 1 名；
- 由于算力紧缺，分组时尽量保证**每组有 1 台具备 8G 显存 GPU 的电脑**；
- 若实在没有本地算力，可自行租用算力平台（见[第八节](#八算力预算与租用)）。

### 1.3 交付物

在一周半的时间内，全组分工协作，完成包括但不限于：

1. 环境搭建
2. 数据收集及清洗
3. 模型结构创新
4. 模型微调训练
5. 模型测试
6. 撰写报告

最后一次课，每组需要进行 **PPT 演示** 并提交一份 **实践报告**。

### 1.4 任务阶段与分工对应

| 阶段 | 具体工作 | 对应本文档 | 主要产出 |
| --- | --- | --- | --- |
| 环境搭建 | 装 torch 与依赖、下载基座模型 | 第三、五节 | 可运行的训练环境 |
| 数据收集及清洗 | 下载数据集、转 `instruction/input/output`、划分 train/val/test | 第四、六节 | `datasets/*.jsonl` |
| 模型结构创新 | 在 LoRA 微调基础上做结构调整或超参改进 | 第七节 | 改进后的训练脚本 |
| 模型微调训练 | 改路径、跑 `train_qwen3.py` | 第七节 | `output/.../final` 适配器 |
| 模型测试 | 跑 `test_qwen3.py`，对比微调前后 | 第七节 | 前后对比 txt、准确率 |
| 撰写报告 | 整理背景、数据、训练、结果与分析 | — | 实践报告 |

### 1.5 报告内容要求

实践报告要详细说明：

- 实践的研究背景
- 数据处理工作
- 模型训练详情
- 实验结果及分析

### 1.6 答辩要求

最后一次课每组进行 **不少于 5 分钟** 的汇报，主要介绍本组本次实践的主要工作以及分工。

### 1.7 注意事项

- 每一位同学都要积极参与，**在实践报告以及答辩 PPT 中均需清楚说明每一位同学的具体贡献**；
- 租赁算力的同学请合理规划算力分配，不要超过给定额度；
- 成绩评定中，**模型性能会纵向对比**（跨组比较），而**报告的规范性、PPT 汇报的效果要横向对比**；
- **不要抄袭！不要抄袭！不要抄袭！** 发现雷同的报告或者模型结果一律按照 0 分计算。

---

## 二、评分规则

**最终成绩 = 出勤情况（10%）+ 实践报告（50%）+ PPT 答辩（20%）+ 模型性能（20%）**

| 项目 | 占比 | 评价方式 |
| --- | --- | --- |
| 出勤情况 | 10% | 集中实验需签到；时间实在冲突时需单独说明情况 |
| 实践报告 | 50% | 考察研究背景、数据处理、模型训练详情、实验结果及分析的完整性（模版统一发放） |
| PPT 答辩 | 20% | 最后一次课汇报，不少于 5 分钟，主要介绍本组主要工作与分工 |
| 模型性能 | 20% | 微调好的模型在**一套全新的数据集**下验证，根据模型测试得分给出成绩 |

**得分要点**：

- 报告占一半权重，重点在「数据处理 + 训练详情 + 结果分析」是否写清楚；
- 模型性能在全新数据集上评测，所以要避免过拟合，训练时的验证集（`DEV_FILE`）就是用来观察是否过拟合的；
- 三个领域都用**准确率**作为评估指标（见[附录数据划分表](#附文件与命令速查)）；
- 分工必须在报告与 PPT 中落实到人。

---

## 三、基准模型的选择

本项目选择 **Qwen3-0.6B** 作为基础模型，可在 ModelScope 的「千问3-0.6B · 模型库」页面下载。

Qwen3 是 Qwen 系列中最新一代的大型语言模型，提供了一系列密集型和混合专家（MoE）模型。

**模型规格**（来自 ModelScope 模型页）：

| 项目 | 值 |
| --- | --- |
| 类型 | 因果语言模型 |
| 训练阶段 | 预训练 & 后训练 |
| 参数数量 | 0.6B |
| 非嵌入参数数量 | 0.44B |
| 层数 | 28 |
| 注意力头数（GQA） | Q 为 16，KV 为 8 |
| 上下文长度 | 32,768 |

**特点**：同一模型内可无缝切换思维模式（适用于复杂逻辑推理、数学和编程）和非思维模式（适用于高效、通用对话）；推理能力在数学、代码生成和常识逻辑推理方面超越之前的 QwQ（思维模式）与 Qwen2.5 指令模型（非思维模式）；在创意写作、角色扮演、多轮对话和指令执行方面的人类偏好对齐更好；支持在思维与非思维模式下与外部工具精确集成；支持 100 多种语言和方言。

### 方式一：命令行下载

1. 打开 pycharm，新建一个项目（例如 `D:\Qwen3-demo`）。
2. 打开终端，安装 ModelScope：

```bash
pip install modelscope
```

3. 下载模型到指定目录：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ［本地模型目录］
```

在模型页面点击「下载模型」后，也可参考面板给出的其它形式：

```bash
# 下载完整模型库
modelscope download --model Qwen/Qwen3-0.6B

# 下载单个文件到指定本地文件夹（以下载 README.md 为例）
modelscope download --model Qwen/Qwen3-0.6B README.md --local_dir ./dir

# Git 下载（需先安装 lfs）
git lfs install
```

### 方式二：SDK 下载

`train_qwen3.py` 中已经写好下载代码，训练运行时模型会自动下载到相应位置：

```python
# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B')
```

---

## 四、数据集下载

三个领域各一个数据集，均在 ModelScope 的「数据集」页找到数据集文件后点击「下载数据集」。

**医学 —— med_qa**（`AI-ModelScope/med_qa`，Apache License 2.0，约 131.78MB）：

```bash
modelscope download --dataset AI-ModelScope/med_qa --local_dir ./datasets/med_qa
```

**法律 —— DISC-Law-SFT**：

```bash
modelscope download --dataset AI-ModelScope/DISC-Law-SFT --local_dir ./datasets/DISC-Law-SFT
```

**金融 —— CFLUE**（金融领域中文语言理解评测数据集）：

```bash
modelscope download --dataset tongyi_dianjin/CFLUE --local_dir ./datasets/CFLUE
```

同模型一样，数据集也支持 SDK 方式：

```python
# 数据集下载
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('AI-ModelScope/med_qa', subset_name='med_qa')
# 您可按需配置 subset_name、split，参照"快速使用"示例代码
```

---

## 五、环境配置

本地使用的是 **RTX5060 8G**，torch 版本为 **2.11.0+cu128**；其它 PyTorch 版本可在官方下载链接里找自己需要的 cuda 版本对应的 torch 下载命令。

```bash
# 安装 torch（cu128）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 六、数据处理

三个领域的处理流程完全一致：把各数据集的选择题转成微调格式 `instruction/input/output`，写入 `datasets/` 下的 jsonl 文件。区别只在于原始数据格式、样本量和划分比例。

### 6.1 医学：`prepare_data.py`

把下载好的数据解压缩到当前文件夹。

**数据内容**：`datasets/med_qa/data_clean/questions/Mainland` 下的 `chinese_qbank.jsonl` 是总的原始数据；`dev.jsonl`、`test.jsonl`、`train.jsonl` 是从原始数据分出的**五选项**训练、测试、验证数据集。`4_options` 文件夹下是**四选项**的训练、测试、验证数据集。本次使用 `4_options` 下的数据集：**训练集 27400 条，验证集 3425 条，测试集 3426 条**。

- **Taiwan** 和 **US** 分别表示台湾和美国的医学数据集，有需要可自行选择。
- **Textbooks** 医学教科书文本语料库是用于检索（RAG）的「知识库」，可以自己扩展。

**格式转换**：把 MedQA 医学选择题（中文，4 选项）转成微调格式。

原始数据格式：

```json
{"question": "卧位腰椎穿刺，脑脊液压力正常值是（　　）。", "options": {"A": "80～180mmH2O（0.78～1.76kPa）", "B": "50～70mmH2O（0.49～0.69kPa）", "C": "230～250mmH2O（2.25～2.45kPa）", "D": "260～280mmH2O（2.55～2.74kPa）"}, "answer": "80～180mmH2O（0.78～1.76kPa）", "meta_info": "第三部分　精神神经系统疾病", "answer_idx": "A"}
```

转换后格式：

```json
{"instruction": "你是一名医学考试助手。请阅读下面的医学单项选择题，从 A、B、C、D 四个选项中选出唯一正确的答案。只输出正确选项的字母和内容，不要解释。", "input": "题目：卧位腰椎穿刺，脑脊液压力正常值是（　　）。\nA. 80～180mmH2O（0.78～1.76kPa）\nB. 50～70mmH2O（0.49～0.69kPa）\nC. 230～250mmH2O（2.25～2.45kPa）\nD. 260～280mmH2O（2.55～2.74kPa）", "output": "A. 80～180mmH2O（0.78～1.76kPa）"}
```

**操作**：

1. 打开项目根目录下的 `prepare_data.py`，修改原始数据路径与输出路径：

```python
# ---- 原始数据路径 ----
RAW_TRAIN = "datasets/med_qa/data_clean/questions/Mainland/4_options/train.jsonl"
RAW_DEV   = "datasets/med_qa/data_clean/questions/Mainland/4_options/dev.jsonl"
RAW_TEST  = "datasets/med_qa/data_clean/questions/Mainland/4_options/test.jsonl"

# ---- 输出路径 ----
OUT_TRAIN = "datasets\\medqa_train.jsonl"
OUT_DEV   = "datasets\\medqa_dev.jsonl"
OUT_TEST  = "datasets\\medqa_test.jsonl"
```

2. 在终端执行：

```bash
python prepare_data.py
```

### 6.2 法律：`prepare_data_law.py`

**数据内容**：`datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv` 是所需数据。

把 DISC-Law-SFT 里的「法律选择题」（**exam 类型**）转成微调格式。原始数据 `law_data/raw/DISC-Law-SFT-Pair.csv` 的列为 `id, input, output`，其中 exam 类型是法律单项选择题（**21,054 条**），所以可以用「准确率」评估。

**划分**：原始数据没有现成 train/val/test，可以自己打乱后切分，**15000 / 500 / 1000**。

**操作**：

1. 打开项目根目录下的 `prepare_data_law.py`，修改路径：

```python
# ---- 原始数据路径 ----
RAW_CSV = "datasets/DISC-Law-SFT/DISC-Law-SFT-Pair.csv"

# ---- 输出路径（放在新建的数据文件夹 law_data 下）----
OUT_TRAIN = "datasets/law_train.jsonl"
OUT_VAL   = "datasets/law_val.jsonl"
OUT_TEST  = "datasets/law_test.jsonl"
```

2. 在终端执行：

```bash
python prepare_data_law.py
```

### 6.3 金融：`prepare_data_finance.py`

把 CFLUE 金融数据集里的「单项选择题」转成微调格式。

**原始数据**：`datasets/CFLUE/{train,val,test}.json`，只取 `task == "单项选择题"`（单选答案是一个字母，能用准确率评估）。

**训练集共 18752 条**，取 **1.5 万**训练，验证集取 **2344**，测试集取 **1000** 条。

**操作**：

1. 打开项目根目录下的 `prepare_data_finance.py`，修改路径：

```python
# ---- 原始数据路径 ----
RAW_TRAIN = "datasets/CFLUE/train.json"
RAW_VAL   = "datasets/CFLUE/val.json"
RAW_TEST  = "datasets/CFLUE/test.json"

# ---- 输出路径 ----
OUT_TRAIN = "datasets/finance_train.jsonl"
OUT_VAL   = "datasets/finance_val.jsonl"
OUT_TEST  = "datasets/finance_test.jsonl"
```

2. 在终端执行：

```bash
python prepare_data_finance.py
```

---

## 七、训练与测试

操作一致，其它数据集运行与其对应的训练与测试文件即可。

### 7.1 训练：`train_qwen3.py`

1. 打开项目根目录下的 `train_qwen3.py`，修改原始数据路径。`MODEL_ID` 是魔搭的模型 id，不用修改。

```python
# ============ 可配置参数 ============
# 注意：ModelScope 上不带 -Instruct 后缀的 Qwen3-0.6B 就是指令版（已做对话训练）
MODEL_ID  = "Qwen/Qwen3-0.6B"
TRAIN_FILE = "datasets/medqa_train.jsonl"   # 训练数据（由 prepare_data.py 生成）
DEV_FILE   = "datasets/medqa_dev.jsonl"     # 验证集，训练时用来评估是否过拟合
OUTPUT_DIR = "./output/Qwen3-0.6B-medqa"
MAX_LENGTH = 512
```

2. 在终端执行：

```bash
python train_qwen3.py
```

### 7.2 测试：`test_qwen3.py`

3. 训练结束后，修改路径，运行测试文件 `test_qwen3.py`。`MODEL_ID` 不用改，会直接生成微调前和微调后的 txt 文件，可以看到训练前后效果。文件生成在根目录下，可以保存在其它位置，给其它数据集让位置。

```python
# ---- 路径 ----
MODEL_ID  = "Qwen/Qwen3-0.6B"
ADAPTER   = "output/Qwen3-0.6B-medqa/final"   # 训练好的 LoRA 适配器
TEST_FILE = "datasets/medqa_test.jsonl"       # 测试集（训练全程没碰过，最干净）
NUM_TEST  = 1000                              # 从测试集里取多少条来测试
```

---

## 八、算力预算与租用

- **预算**：一组用于租算力的金额**不得超过 200 元**；课程结束后凭**订单截图及发票**报销。
- **不需要买贵卡**：需要训练的模型并不大，在 **8G 显存 GPU** 上可以正常运行，优先选择性价比较高的显卡（例如 **4080、5060** 这类），不要一上来就选价格昂贵的显卡；选择之前先查资料，学会挑合适的 GPU 型号。
- **可用平台**：没有强制要求，可以考虑 autoDL、算家云、优云智算、晨涧云等。
- **省钱技巧**：注意云服务器收费规则——有些供应商只要上传了数据，即使没运行程序也会计时收费。合理规划训练时间与效果：**调试代码放在收费低的显卡上跑，实际训练再换成稍高端一些的显卡**。
- 本着节俭节约的精神合理充值账号，并保留订单、付款、发票等信息便于后续核对。

---

## 附：文件与命令速查

| 步骤 | 内容 | 关键命令 / 文件 |
| --- | --- | --- |
| 三 | 基准模型选择 Qwen3-0.6B | `modelscope download --model Qwen/Qwen3-0.6B --local_dir ［目标目录］`；或 `train_qwen3.py` 内置 SDK 下载 |
| 四 | 数据集下载（医学 / 法律 / 金融） | `modelscope download --dataset AI-ModelScope/med_qa`、`AI-ModelScope/DISC-Law-SFT`、`tongyi_dianjin/CFLUE` |
| 五 | 环境配置（RTX5060 8G，torch 2.11.0+cu128） | `pip install torch --index-url https://download.pytorch.org/whl/cu128`；`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 六 | 数据处理（转 instruction/input/output） | `prepare_data.py`（医学）、`prepare_data_law.py`（法律）、`prepare_data_finance.py`（金融） |
| 七 | 训练与测试 | `train_qwen3.py`、`test_qwen3.py`（生成微调前后对比 txt） |
| 八 | 算力与预算 | 8G 显存即可；优先 4080、5060；租赁上限 200 元/组；低价卡调代码、高价卡跑训练 |

**数据集与划分一览**

| 领域 | 数据集 | 原始数据 | 训练 / 验证 / 测试 | 评估指标 |
| --- | --- | --- | --- | --- |
| 医学 | med_qa（4_options） | `Mainland/4_options/{train,dev,test}.jsonl` | 27400 / 3425 / 3426 | 准确率 |
| 法律 | DISC-Law-SFT（exam） | `DISC-Law-SFT-Pair.csv`（21054 条） | 15000 / 500 / 1000 | 准确率 |
| 金融 | CFLUE（单项选择题） | `CFLUE/{train,val,test}.json`（18752 条） | 15000 / 2344 / 1000 | 准确率 |

**评分要点速查**

| 项目 | 占比 | 一句话要求 |
| --- | --- | --- |
| 出勤情况 | 10% | 集中实验签到，冲突需单独说明 |
| 实践报告 | 50% | 背景 + 数据处理 + 训练详情 + 结果分析，分工写到人 |
| PPT 答辩 | 20% | 最后一次课，不少于 5 分钟 |
| 模型性能 | 20% | 在全新数据集上验证，按测试得分给成绩 |

**训练/测试产物**

- `./output/Qwen3-0.6B-medqa/final`：训练好的 LoRA 适配器（`ADAPTER` 指向这里）
- 根目录下的微调前后对比 txt：由 `test_qwen3.py` 生成，可移走以给其它数据集腾位置
