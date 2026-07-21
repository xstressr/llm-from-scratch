"""Plot tanh and its local gradient 1 - tanh^2.

Run from repo root (venv activated):
    python 02-makemore/plot_tanh.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "tanh_saturation.png"

x = np.linspace(-5, 5, 400)
y = np.tanh(x)
grad = 1 - y**2

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(x, y, color="#2563eb", lw=2, label="tanh(x)")
ax.axhline(1, color="gray", ls="--", lw=0.8)
ax.axhline(-1, color="gray", ls="--", lw=0.8)
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.fill_between(
    x, y, where=(np.abs(y) > 0.9), color="#ef4444", alpha=0.25, label="|tanh|~1 saturated"
)
ax.set_xlabel("preact x")
ax.set_ylabel("h = tanh(x)")
ax.set_title("tanh squeezes to (-1, 1)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(-1.3, 1.3)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x, grad, color="#16a34a", lw=2, label=r"local grad $1-\tanh(x)^2$")
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("preact x")
ax.set_ylabel("d(tanh)/dx")
ax.set_title("grad ~0 at tails, ~1 near 0")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
