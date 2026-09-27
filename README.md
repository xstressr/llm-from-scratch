# llm-from-scratch

按 Karpathy 路线从零理解深度学习与 LLM：自动微分 → 字符级语言模型 → GPT 训练 → C 底层实现 → 最小 Chat 栈。本仓库只放可跑代码与实验，长文写在博客。

## 阶段

| 目录 | 阶段 | 状态 |
|------|------|------|
| `01-micrograd/` | 手写标量 autograd + 最小 MLP | 已完成 |
| `02-makemore/` | 字符级 LM：bigram → MLP → WaveNet 风格层次融合 | 已完成（test NLL ≈ 2.10） |
| `03-nanogpt/` | 最小 GPT 训练循环 + 受控对照实验 | 已完成（工程 baseline + 三组对照 + 加长训） |
| `04-c-impl/` | `llama2.c` / `llm.c`：读调用链、编译、训练 | 已完成（结果见下） |
| `05-nanochat/` | tokenizer → 预训练 → SFT → 推理服务 | 暂停（2026-09），未开始 |

## 结果

`04-c-impl/`，单卡 RTX 3060（6 GB）：

- **llama2.c 15M × TinyStories**：从零训练 260K 步（约 10.3 h），终点 val loss **1.1018**（官方 stories15M 为 1.072）；`run.c` 推理速度与官方权重持平（约 352 tok/s，8 线程）。
- **llm.c GPT-2 124M**：CPU 版 `train_gpt2.c`、FP32 CUDA `train_gpt2fp32cu`、BF16 混合精度 `train_gpt2cu` 均编译跑通（6 GB 显存需 `-b 4 -t 64`）。

曲线脚本：`04-c-impl/plot_llama2_260k_loss.py`。

## 相关文章

Zero to One AI 学习日志（[blog.crazyai.uk](https://blog.crazyai.uk/)）：

- [The Batch Is a Factory of Futures](https://blog.crazyai.uk/posts/transformer-field-notes-01-batch-factory-of-futures/)（Transformer Field Notes 系列第 1 篇）
- [Python Writes the Answer Sheet](https://blog.crazyai.uk/posts/c-field-notes-01-python-writes-the-answer-sheet/)（llm.c）
- [There Is No Autograd Here](https://blog.crazyai.uk/posts/c-field-notes-02-there-is-no-autograd-here/)（llm.c）
- [The Model File Has No Names](https://blog.crazyai.uk/posts/inference-field-notes-01-the-model-file-has-no-names/)（llama2.c 推理）

## 本地 vs Colab（VS Code 扩展）

代码始终在本仓库本地编辑。需要 GPU 时，用 VS Code 的 **Colab** 扩展连接远端 runtime，直接在本机打开的 notebook / 终端里跑（无需单独维护上传版 Colab notebook）。

| 工作 | 在哪跑 |
|------|--------|
| 手写 / 读代码 / 小实验（micrograd、makemore、C） | 本地 CPU |
| nanoGPT / nanochat 等 GPU 训练 | 本机打开文件 → Colab 扩展连远端 GPU |
| llama2.c / llm.c CUDA 训练 | WSL2 + CUDA（见 `04-c-impl/README.md`） |

checkpoint 不提交 git（`runs/`、`out/` 已 ignore）。

## 环境

本仓库用 [uv](https://docs.astral.sh/uv/) 管 Python 和依赖，不要再用 `python -m venv` + `pip install`。

```bash
uv sync --extra torch
```

之后用 `uv run` 跑脚本（会自动走仓库根的 `.venv`），例如：

```bash
uv run python 01-micrograd/train_toy.py
```

如果要在当前终端里激活解释器：

```powershell
.\.venv\Scripts\Activate.ps1
```

WSL 脚本的路径可用环境变量覆盖：`LLM_SCRATCH_PY`（解释器，默认 `~/venvs/llm-scratch/bin/python`）、`LLAMA2_LOG`（训练日志）、`TINYSTORIES_DIR`（数据解压目录）。

## 快速开始

```bash
# 阶段 1
uv run python 01-micrograd/train_toy.py

# 阶段 2
uv run python 02-makemore/bigram.py

# 阶段 3：用 VS Code 打开 baseline Notebook 并连接 Colab
code 03-nanogpt/experiments/01_tiny_shakespeare_baseline.ipynb
```

## 约定

- **练习代码**：写在各阶段目录根下（如 `value.py`）。上游仓库放在各阶段的 `vendor/`，只在本地 clone，不提交。
- **草稿**：`notes/` / `call-chains/` / `deploy-notes/` 只作草稿，成文写到博客。
- **产物**：checkpoint、大数据、`.venv` 一律 gitignore。

## 致谢与许可

学习路线与参考实现来自 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)、[makemore](https://github.com/karpathy/makemore)、[nanoGPT](https://github.com/karpathy/nanoGPT)、[llama2.c](https://github.com/karpathy/llama2.c)、[llm.c](https://github.com/karpathy/llm.c)、[nanochat](https://github.com/karpathy/nanochat)（均为 MIT）。本仓库代码以 [MIT](LICENSE) 许可发布。
