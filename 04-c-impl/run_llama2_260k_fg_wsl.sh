#!/usr/bin/env bash
# Foreground 260K train (keep this WSL session alive — nohup dies when wsl.exe exits).
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:/home/xstre/.local/bin"
export PYTHONUNBUFFERED=1
LOG="/home/xstre/llama2_260k.log"
echo "260K_START $(date -Iseconds)" | tee "$LOG"
/usr/bin/bash /mnt/d/Projects/llm-from-scratch/04-c-impl/train_llama2_wsl.sh \
  --out_dir=out \
  --max_iters=260000 \
  --eval_interval=2000 \
  --eval_iters=10 \
  --always_save_checkpoint=True \
  --init_from=scratch \
  2>&1 | tee -a "$LOG"
echo "260K_END $(date -Iseconds) exit=${PIPESTATUS[0]}" | tee -a "$LOG"
