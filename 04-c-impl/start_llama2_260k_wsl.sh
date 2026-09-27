#!/usr/bin/env bash
# Start official-scale 15M TinyStories train (max_iters=260000).
# Do NOT rely on `wsl.exe ... nohup &` — the job dies when that wsl.exe exits.
# Keep a living WSL session: bash run_llama2_260k_fg_wsl.sh
# Resume: train_llama2_wsl.sh --out_dir=out --init_from=resume --max_iters=260000
# Monitor: tail -f $HOME/llama2_260k.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
LOG="${LLAMA2_LOG:-$HOME/llama2_260k.log}"
PIDF="${LOG%.log}.pid"
LLAMA="$HERE/vendor/llama2.c"
cd "$LLAMA"
nohup /usr/bin/bash "$HERE/train_llama2_wsl.sh" \
  --out_dir=out \
  --max_iters=260000 \
  --eval_interval=2000 \
  --eval_iters=10 \
  --always_save_checkpoint=True \
  --init_from=scratch \
  > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") log=$LOG"
