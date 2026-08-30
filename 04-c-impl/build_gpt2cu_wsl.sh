#!/usr/bin/env bash
# WSL2 Ubuntu + CUDA toolkit at /usr/local/cuda (often not on PATH).
# Native Windows has no nvcc. Does not run make clean (keeps train_gpt2fp32cu).
set -euo pipefail
export PATH="/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib${PATH:+:$PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$(dirname "$0")/vendor/llm.c"
# RTX 3060 = sm_86. Default PRECISION=BF16 needs gpt2_124M_bf16.bin.
# Run: ./train_gpt2cu -b 4 -t 64   (source default T=1024 OOMs on 6GB)
make train_gpt2cu GPU_COMPUTE_CAPABILITY=86
