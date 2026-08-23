# llm-from-scratch

按 Karpathy 路线从零理解深度学习与 LLM：自动微分 → 字符级语言模型 → GPT 训练 → C 底层实现 → 最小 Chat 栈。

正式笔记在 Obsidian Vault：`01-Learning/AI-ML/Karpathy学习路线.md`。本仓库只放可跑代码与实验。

## 阶段

| 目录 | 阶段 | 状态 |
|------|------|------|
| `01-micrograd/` | 手写标量 autograd + 最小 MLP | 已完成 |
| `02-makemore/` | 字符级 LM：bigram → MLP → WaveNet 风格层次融合 | 主干已完成（test NLL ≈ 2.10） |
| `03-nanogpt/` | 跑通小规模 GPT 训练与受控对照实验 | 下一阶段；上游放 `vendor/` |
| `04-c-impl/` | `llama2.c` / `llm.c` 调用链阅读 | 目录占位 |
| `05-nanochat/` | tokenizer → 训练 → 推理服务端到端 | 目录占位 |

上游仓库暂不 clone；需要时再放进各阶段的 `vendor/`（或改成 git submodule）。


## 本地 vs Colab（VS Code 扩展）

代码始终在本仓库本地编辑。需要 GPU 时，用 VS Code 的 **Colab** 扩展连接远端 runtime，直接在本机打开的 notebook / 终端里跑（无需单独维护上传版 Colab notebook）。

| 工作 | 在哪跑 |
|------|--------|
| 手写 / 读代码 / 小实验（micrograd、makemore、C） | 本地 CPU |
| nanoGPT / nanochat 等 GPU 训练 | 本机打开文件 → Colab 扩展连远端 GPU |

checkpoint 仍不要提交 git（`runs/` 已 ignore）。

## 环境

本仓库用 [uv](https://docs.astral.sh/uv/) 管 Python 和依赖，不要再用 `python -m venv` + `pip install`。

```bash
cd D:\Projects\llm-from-scratch
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

## 快速开始

```bash
# 阶段 1
uv run python 01-micrograd/train_toy.py

# 阶段 2（已完成，可随时复跑）
uv run python 02-makemore/bigram.py

# 阶段 3（当前下一步）：用 VS Code 打开 baseline Notebook 并连接 Colab
code 03-nanogpt/experiments/01_tiny_shakespeare_baseline.ipynb
```


## Agent / 笔记分流

本仓库只放代码。Agent 规则见 [`AGENTS.md`](AGENTS.md)。

正式笔记写到 Vault：`01-Learning/AI-ML/`。可用 [`llm-from-scratch.code-workspace`](llm-from-scratch.code-workspace) 在同一窗口打开「代码 + Vault」。

## 约定

- **练习代码**：写在各阶段目录根下（如 `value.py`），不要改 `vendor/` 里的上游。
- **笔记**：结论写回 Vault；这里的 `notes/` / `call-chains/` / `deploy-notes/` 只作草稿。
- **产物**：checkpoint、大数据、`.venv` 一律 gitignore。
