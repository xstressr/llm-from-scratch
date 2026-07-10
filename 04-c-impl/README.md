# 04 · llama2.c / llm.c

目标：从工程角度理解推理与训练——权重加载、前向、采样、反向、优化器更新。

## 目录

- `vendor/` — `llama2.c` / `llm.c`（暂未 clone）
- `call-chains/` — 每次只追一条路径，整理关键函数调用链

学习方式：不要求一次读完；每次只追「token → logits」或「loss → 参数更新」之一。
