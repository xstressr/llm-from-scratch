"""Parse a llama2.c train.py log and write loss-curve PNGs.

Usage: python plot_llama2_260k_loss.py [LOG] [OUT_DIR]
  LOG defaults to ./llama2_260k.log, OUT_DIR to 04-c-impl/notes/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("llama2_260k.log")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "notes"

TRAIN_RE = re.compile(r"^(\d+) \| loss ([\d.]+)")
VAL_RE = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)")

RAW_STRIDE = 50
SMOOTH_WINDOW = 200
SMOOTH_STRIDE = 10


def parse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_s: list[int] = []
    train_l: list[float] = []
    val_s: list[int] = []
    val_l: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TRAIN_RE.match(line)
            if m:
                train_s.append(int(m.group(1)))
                train_l.append(float(m.group(2)))
                continue
            m = VAL_RE.match(line)
            if m:
                val_s.append(int(m.group(1)))
                val_l.append(float(m.group(3)))
    return (
        np.asarray(train_s, dtype=np.int32),
        np.asarray(train_l, dtype=np.float32),
        np.asarray(val_s, dtype=np.int32),
        np.asarray(val_l, dtype=np.float32),
    )


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Causal-ish trailing mean with a shrinking window at the start (no zero-pad)."""
    if window < 2 or y.size < 2:
        return y.astype(np.float64, copy=True)
    c = np.cumsum(np.insert(y.astype(np.float64), 0, 0.0))
    n = y.size
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


def style_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_xlim(0, 260_000)
    ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000, 260_000])
    ax.set_xticklabels(["0", "50k", "100k", "150k", "200k", "260k"])
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#888888")


def save(fig, name: str) -> Path:
    dest = OUT / name
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def main() -> None:
    ts, tl, vs, vl = parse(LOG)
    print(f"parsed train={ts.size} val={vs.size}")
    if ts.size:
        print(f"train step {int(ts[0])}..{int(ts[-1])} loss {tl[0]:.4f}..{tl[-1]:.4f}")
    if vs.size:
        print(f"val step {int(vs[0])}..{int(vs[-1])} loss {vl[0]:.4f}..{vl[-1]:.4f}")

    raw_s = ts[::RAW_STRIDE]
    raw_l = tl[::RAW_STRIDE]
    smooth = rolling_mean(tl, SMOOTH_WINDOW)
    sm_s = ts[::SMOOTH_STRIDE]
    sm_l = smooth[::SMOOTH_STRIDE]

    title_base = "15M TinyStories · RTX 3060 · 2026-08-30"

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="white")
    ax.plot(raw_s, raw_l, color="#7aa6d4", linewidth=0.5, alpha=0.45, label=f"per-step (every {RAW_STRIDE})")
    ax.plot(sm_s, sm_l, color="#1f4e79", linewidth=1.6, label=f"rolling mean ({SMOOTH_WINDOW})")
    style_axes(ax, f"Train loss · {title_base}")
    ax.legend(frameon=False, loc="upper right")
    p1 = save(fig, "llama2-15M-260K-train-loss.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="white")
    ax.plot(vs, vl, color="#c45c26", linewidth=1.4, marker="o", markersize=3.5, label="val (every 2000 steps)")
    style_axes(ax, f"Val loss · {title_base}")
    ax.legend(frameon=False, loc="upper right")
    p2 = save(fig, "llama2-15M-260K-val-loss.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="white")
    ax.plot(raw_s, raw_l, color="#7aa6d4", linewidth=0.45, alpha=0.35, label=f"train per-step (every {RAW_STRIDE})")
    ax.plot(sm_s, sm_l, color="#1f4e79", linewidth=1.5, label=f"train rolling mean ({SMOOTH_WINDOW})")
    ax.plot(vs, vl, color="#c45c26", linewidth=1.5, marker="o", markersize=4, label="val (every 2000)")
    style_axes(ax, f"Train + val loss · {title_base}")
    ax.legend(frameon=False, loc="upper right")
    p3 = save(fig, "llama2-15M-260K-loss-curves.png")

    print("wrote", p1)
    print("wrote", p2)
    print("wrote", p3)
    print(f"plotted train_raw={raw_s.size} train_smooth={sm_s.size} val={vs.size}")
    print(f"source_train={ts.size} source_val={vs.size}")


if __name__ == "__main__":
    main()
