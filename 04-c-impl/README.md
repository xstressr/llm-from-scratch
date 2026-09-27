# 04 · llama2.c / llm.c

目标：从工程角度理解推理与训练——权重加载、前向、采样、反向、优化器更新。

## 目录

- `vendor/` — `llama2.c` / `llm.c`（本机 2026-08-24 已 clone `llm.c`；gitignore 本地 checkout，非 submodule）
- `call-chains/` — 每次只追一条路径，整理关键函数调用链

学习方式：不要求一次读完；每次只追「token → logits」或「loss → 参数更新」之一。正式笔记在 Vault `01-Learning/AI-ML/`（架构四件套 / run.c / export.py），此处不写长文。

Windows CPU 编译（MSVC）：`build_cpu_msvc.bat`。CUDA 走 WSL2 + `/usr/local/cuda`（原生 Windows 无 nvcc）；**必须** `-b 4 -t 64`。

- FP32：`make train_gpt2fp32cu GPU_COMPUTE_CAPABILITY=86` → `./train_gpt2fp32cu`
- 主线 BF16：`04-c-impl/build_gpt2cu_wsl.sh` 或 `make train_gpt2cu GPU_COMPUTE_CAPABILITY=86` → `./train_gpt2cu`（需要 `gpt2_124M_bf16.bin`）
- llama2 训练：`setup_llama2_wsl.sh` / `train_llama2_wsl.sh` / `start_llama2_260k_wsl.sh`；解释器是 WSL venv（默认 `~/venvs/llm-scratch`，CUDA torch，可用 `LLM_SCRATCH_PY` 覆盖），不是 Windows CPU `.venv`

数字写回 Vault `01-Learning/AI-ML/`。
