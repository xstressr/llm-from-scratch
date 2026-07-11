"""Scalar autograd engine (micrograd-style).

Learning task: fill in the TODOs until `python train_toy.py` trains a tiny MLP.
Reference shape: https://github.com/karpathy/micrograd
"""

from __future__ import annotations

from typing import Callable, Set, Tuple, Union
import math

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
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward() -> None:
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: Number) -> "Value":
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f"**{exponent}")
        def _backward() -> None:
            # d(out)/d(self) = exponent * self.data^(exponent-1)
            self.grad += exponent * self.data ** (exponent-1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward() -> None:
            # d(out)/d(self) = 1 - tanh(self.data)^2
            self.grad += (1 - math.tanh(self.data)**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0, self.data), (self,), "relu")
        def _backward() -> None:
            # d(out)/d(self) = 1 if self.data > 0 else 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward() -> None:
            # d(out)/d(self) = exp(self.data)
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward
        return out

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
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        topo.reverse()
        for node in topo:
            node._backward()


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
