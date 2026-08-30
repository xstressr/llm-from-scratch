#!/usr/bin/env bash
# WSL helper: deps + unpack TinyStories onto ext4 (not NTFS) + pretok.
# Avoid running this via PowerShell double-quoted $vars; execute the file.
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:/home/xstre/.local/bin:/usr/local/bin"
LLAMA="/mnt/d/Projects/llm-from-scratch/04-c-impl/vendor/llama2.c"
PY="/home/xstre/venvs/llm-scratch/bin/python"
DEST="/home/xstre/tinystories/TinyStories_all_data"
TAR="${LLAMA}/data/TinyStories_all_data.tar.gz"

echo "== deps =="
uv pip install --python "$PY" numpy sentencepiece requests tqdm
"$PY" -c "import numpy,sentencepiece,torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== unpack =="
mkdir -p "$DEST"
if [ -z "$(ls -A "$DEST" 2>/dev/null | head -1)" ]; then
  tar -xzf "$TAR" -C "$DEST"
fi
echo "shards: $(ls "$DEST"/*.json 2>/dev/null | wc -l)"
ln -sfn "$DEST" "${LLAMA}/data/TinyStories_all_data"

echo "== pretok (llama2 tokenizer) =="
cd "$LLAMA"
# Staged: first 3 shards so train can start; remainder via tinystories.py later.
"$PY" - <<'PY'
from tinystories import process_shard
import glob, os
shards = sorted(glob.glob("data/TinyStories_all_data/*.json"))
print(f"json shards: {len(shards)}")
n = min(3, len(shards))
for i, s in enumerate(shards[:n]):
    binp = s.replace(".json", ".bin")
    if os.path.exists(binp):
        print(f"skip existing {binp}")
        continue
    print(f"pretok {i}: {s}")
    process_shard((i, s), vocab_size=0)
print("quick pretok done")
PY
echo "bin shards now: $(ls "$DEST"/*.bin 2>/dev/null | wc -l)"
echo "SETUP_DONE"
