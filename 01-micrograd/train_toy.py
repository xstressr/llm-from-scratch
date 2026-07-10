#!/usr/bin/env python3
"""Toy binary classification with a tiny MLP.

Fill in value.py / nn.py, then re-run this file.
"""

from __future__ import annotations

import sys

from value import Value, is_implemented


def main() -> int:
    if not is_implemented():
        print("01-micrograd scaffold is ready, but Value is still a stub.")
        print()
        print("Next steps:")
        print("  1. Implement Value.__add__ / __mul__ / __pow__ / tanh (or relu)")
        print("  2. Implement Value.backward (topo sort + local grads)")
        print("  3. Implement Neuron.__call__ in nn.py")
        print("  4. Re-run: python train_toy.py")
        print()
        print("Vault note: 01-Learning/AI-ML/Karpathy学习路线.md → 第 1 阶段")
        return 0

    from nn import MLP

    # classic micrograd moon-ish toy: 4 points, 2 classes
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])
    print(f"parameters: {len(model.parameters())}")

    for step in range(50):
        # forward
        ypred = [model([Value(v) for v in x]) for x in xs]
        loss = sum((yout - yref) ** 2 for yout, yref in zip(ypred, ys))  # type: ignore[arg-type]
        assert isinstance(loss, Value)

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data -= 0.05 * p.grad

        if step % 10 == 0 or step == 49:
            preds = [round(yp.data, 3) if isinstance(yp, Value) else yp for yp in ypred]
            print(f"step {step:02d}  loss={loss.data:.4f}  preds={preds}")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
