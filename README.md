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
   modelscope download --model Qwen/Qwen3-0.6B --local_dir ./qwen3
   ```
