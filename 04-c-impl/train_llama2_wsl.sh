#!/usr/bin/env bash
# llama2.c train.py on WSL CUDA torch. Sequential GPU: do not run beside llm.c CUDA training.
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:/home/xstre/.local/bin"
export PYTHONUNBUFFERED=1
LLAMA="/mnt/d/Projects/llm-from-scratch/04-c-impl/vendor/llama2.c"
PY="/home/xstre/venvs/llm-scratch/bin/python"
cd "$LLAMA"
exec "$PY" train.py \
  --compile=False \
  --dtype=float16 \
  --batch_size=8 \
  --max_seq_len=256 \
  --gradient_accumulation_steps=4 \
  --eval_iters=10 \
  "$@"
