# 03 · nanoGPT

目标：看懂 GPT 训练循环（tokenizer、dataset、attention、block、optimizer），并通过可重复实验建立模型规模、训练成本与效果之间的直觉。

当前里程碑：**Tiny Shakespeare baseline 已就绪，等待在 VS Code Colab runtime 上执行完整训练。**

## 目录

| 路径 | 作用 |
|---|---|
| `data.py` | 下载语料、字符级 tokenizer、train/val split、batch sampling |
| `model.py` | 最小 GPT：causal self-attention、MLP、Transformer block、生成 |
| `train.py` | 训练、评估、指标记录、checkpoint 保存与恢复 |
| `configs/tiny_shakespeare_baseline.json` | 第一组固定 baseline 参数 |
| `experiments/01_tiny_shakespeare_baseline.ipynb` | 工程 baseline 实验控制台 |
| `experiments/02_karpathy_lets_build_gpt.ipynb` | 课堂版：跟着 Lecture 7 从 bigram 搭到 GPT |
| `data/` | 本地/远端数据缓存（gitignore） |
| `runs/` | checkpoint 与实验指标（gitignore） |
| `vendor/` | 后续放置 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 供逐段对照 |

## 第一次实验

若在跟 Karpathy 视频，优先打开课堂版：

```text
experiments/02_karpathy_lets_build_gpt.ipynb
```

官方原版 Colab：[Let's build GPT](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)（视频描述同款；Lecture 7 未收入 GitHub lectures/）。

工程对照实验用：

```text
experiments/01_tiny_shakespeare_baseline.ipynb
```

连接 Colab runtime 后按顺序执行。Notebook **自包含**数据工具 / 模型 / 训练代码，不依赖本机私有仓库同步；唯一外部文件是公开语料 `tinyshakespeare.txt`（自动下载到 `data/`）。本地的 `data.py` / `model.py` / `train.py` 仍供 CLI smoke test 与对照阅读。

Notebook 会完成：

1. 检查远端 Python、PyTorch、CUDA、GPU 与工作目录。
2. 在 notebook 内定义代码，并下载 Tiny Shakespeare。
3. 在固定 batch 上过拟合，验证 forward/backward 链路。
4. 运行 4 层、4 heads、128 embedding 的 baseline。
5. 绘制 train/val loss，并从固定 `ROMEO:` prompt 采样。
6. 重新加载 checkpoint，比较恢复前后的 logits。

默认产物写入 `runs/baseline/`。Colab runtime 文件可能随实例释放而消失；如需持久保存，运行 baseline 前设置：

```python
import os
os.environ["NANOGPT_RUN_DIR"] = "/path/to/persistent/storage/nanogpt-baseline"
```

## 本地 smoke test

```bash
cd 03-nanogpt
../.venv/bin/python train.py --smoke --device cpu
```

这只运行 5 个训练 step，用于验证数据下载、训练、指标与 checkpoint 链路，不代表模型效果。

## 实验纪律

- Notebook 负责假设、配置、调用、可视化与短结论；可复用逻辑留在 `.py`。
- 一次只改变一个主要变量；baseline 固定 `seed=42`。
- 至少记录参数量、训练 token、train/val loss、tokens/sec、最大显存、训练时间和固定 prompt 样本。
- checkpoint、大语料和 `runs/` 不提交 Git；验证后的学习结论整理到 Vault。

后续按 `block_size` → `n_layer` → `n_embd` 的顺序做三组对照，再进入上游 nanoGPT 调用链阅读。
