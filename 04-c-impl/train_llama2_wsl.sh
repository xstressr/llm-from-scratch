#!/usr/bin/env bash
# llama2.c train.py on WSL CUDA torch. Sequential GPU: do not run beside llm.c CUDA training.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1
LLAMA="$HERE/vendor/llama2.c"
PY="${LLM_SCRATCH_PY:-$HOME/venvs/llm-scratch/bin/python}"
cd "$LLAMA"
exec "$PY" train.py \
  --compile=False \
  --dtype=float16 \
  --batch_size=8 \
  --max_seq_len=256 \
  --gradient_accumulation_steps=4 \
  --eval_iters=10 \
  "$@"
