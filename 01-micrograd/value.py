"""Scalar autograd engine (micrograd-style).

Learning task: fill in the TODOs until `python train_toy.py` trains a tiny MLP.
Reference shape: https://github.com/karpathy/micrograd
"""

from __future__ import annotations

from typing import Callable, Set, Tuple, Union

Number = Union[int, float]


class Value:
    """A scalar node in a computational graph."""

    def __init__(
        self,
        data: Number,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        # local backward: accumulate grads into children
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- arithmetic (implement these) ---

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        # TODO: out = Value(self.data + other.data, (self, other), "+")
        # TODO: define out._backward to distribute out.grad to self/other
        raise NotImplementedError("implement Value.__add__")

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        # TODO: product rule for grads
        raise NotImplementedError("implement Value.__mul__")

    def __pow__(self, exponent: Number) -> "Value":
        assert isinstance(exponent, (int, float))
        # TODO: d/dx x^n = n * x^(n-1)
        raise NotImplementedError("implement Value.__pow__")

    def tanh(self) -> "Value":
        # TODO: out = tanh(self.data); grad *= (1 - t^2)
        raise NotImplementedError("implement Value.tanh")

    def relu(self) -> "Value":
        # TODO: out = max(0, self.data); grad *= (self.data > 0)
        raise NotImplementedError("implement Value.relu")

    def exp(self) -> "Value":
        raise NotImplementedError("implement Value.exp (optional)")

    # --- sugar ---

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-other)

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        return self * (other**-1 if isinstance(other, Value) else Value(other) ** -1)

    def __radd__(self, other: Number) -> "Value":
        return self + other

    def __rmul__(self, other: Number) -> "Value":
        return self * other

    def __rsub__(self, other: Number) -> "Value":
        return Value(other) + (-self)

    def __rtruediv__(self, other: Number) -> "Value":
        return Value(other) * (self**-1)

    # --- backward ---

    def backward(self) -> None:
        """Reverse-mode AD: topo-sort then apply local `_backward`."""
        # TODO:
        # 1. topological sort of all nodes reachable from self via _prev
        # 2. self.grad = 1.0
        # 3. for each node in reverse topo order: node._backward()
        raise NotImplementedError("implement Value.backward")


def is_implemented() -> bool:
    """Smoke check used by train_toy.py — true once core ops work."""
    try:
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b + Value(1.0)
        d = c.tanh()
        d.backward()
        return abs(a.grad) > 0 or abs(b.grad) > 0
    except NotImplementedError:
        return False
