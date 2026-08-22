# 04 · llama2.c / llm.c

目标：从工程角度理解推理与训练——权重加载、前向、采样、反向、优化器更新。

## 目录

- `vendor/` — `llama2.c` / `llm.c`（2026-08-22 已 clone；gitignore 本地 checkout，非 submodule）
- `call-chains/` — 每次只追一条路径，整理关键函数调用链

学习方式：不要求一次读完；每次只追「token → logits」或「loss → 参数更新」之一。正式笔记在 Vault `01-Learning/AI-ML/`（架构四件套 / run.c / export.py），此处不写长文。
