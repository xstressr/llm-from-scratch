#!/usr/bin/env bash
# Pretokenize remaining TinyStories shards (llama2 tokenizer). CPU only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/sbin:/usr/bin:/sbin:/bin"
LLAMA="$HERE/vendor/llama2.c"
PY="${LLM_SCRATCH_PY:-$HOME/venvs/llm-scratch/bin/python}"
cd "$LLAMA"
"$PY" - <<'PY'
from tinystories import process_shard
import glob, os
shards = sorted(glob.glob("data/TinyStories_all_data/*.json"))
print(f"json shards: {len(shards)}")
for i, s in enumerate(shards):
    binp = s.replace(".json", ".bin")
    if os.path.exists(binp):
        continue
    print(f"pretok {i}: {os.path.basename(s)}")
    process_shard((i, s), vocab_size=0)
print("PRETOK_ALL_DONE", sum(1 for s in shards if os.path.exists(s.replace('.json','.bin'))))
PY
