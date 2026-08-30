#!/usr/bin/env bash
# Start official-scale 15M TinyStories train (max_iters=260000).
# Do NOT rely on `wsl.exe ... nohup &` — the job dies when that wsl.exe exits.
# Keep a living WSL session: bash run_llama2_260k_fg_wsl.sh
# Resume: train_llama2_wsl.sh --out_dir=out --init_from=resume --max_iters=260000
# Monitor: tail -f /home/xstre/llama2_260k.log
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:/home/xstre/.local/bin"
LOG="/home/xstre/llama2_260k.log"
PIDF="/home/xstre/llama2_260k.pid"
LLAMA="/mnt/d/Projects/llm-from-scratch/04-c-impl/vendor/llama2.c"
cd "$LLAMA"
nohup /usr/bin/bash /mnt/d/Projects/llm-from-scratch/04-c-impl/train_llama2_wsl.sh \
  --out_dir=out \
  --max_iters=260000 \
  --eval_interval=2000 \
  --eval_iters=10 \
  --always_save_checkpoint=True \
  --init_from=scratch \
  > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") log=$LOG"
