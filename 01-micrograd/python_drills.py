#!/usr/bin/env python3
"""Python 语法实操：对照 nn.py 里出现的写法。

用法：
  cd 01-micrograd
  python python_drills.py          # 跑全部题，看对错
  python python_drills.py 1        # 只跑第 1 题（写 1 或 01 都行）

每题把 `TODO` 换成你的实现；不要偷看下面的 reference_*（那是答案）。
"""

from __future__ import annotations

import re
import sys
from typing import Any, Callable, List


# ---------------------------------------------------------------------------
# 第 1 题：列表推导 —— 对应 Neuron 里建权重
#   self.w = [Value(...) for _ in range(nin)]
# ---------------------------------------------------------------------------

def drill_01() -> List[int]:
    """把 [1, 2, 3, 4] 变成每个数的平方。用一行列表推导，不要写 for 循环块。"""
    nums = [1, 2, 3, 4]
    return [n * n for n in nums]


def reference_01() -> List[int]:
    return [n * n for n in [1, 2, 3, 4]]


# ---------------------------------------------------------------------------
# 第 2 题：zip —— 对应 Neuron 里 wi * xi
#   sum((wi * xi for wi, xi in zip(self.w, x)), ...)
# ---------------------------------------------------------------------------

def drill_02() -> List[int]:
    """把 w 和 x 两两相乘，返回乘积列表。必须用 zip。"""
    w = [2, 3, 4]
    x = [10, 100, 1000]
    return [wi * xi for wi, xi in zip(w, x)]


def reference_02() -> List[int]:
    w = [2, 3, 4]
    x = [10, 100, 1000]
    return [wi * xi for wi, xi in zip(w, x)]


# ---------------------------------------------------------------------------
# 第 3 题：生成器表达式 + sum —— 对应加权和
#   sum((wi * xi for wi, xi in zip(...)), 起点)
# 注意：(...) 是生成器；[...] 是列表。sum 两种都能吃。
# ---------------------------------------------------------------------------

def drill_03() -> int:
    """计算 w·x = 2*10 + 3*100 + 4*1000，用 sum + 生成器表达式（圆括号）。"""
    w = [2, 3, 4]
    x = [10, 100, 1000]
    return sum((wi * xi for wi, xi in zip(w, x)), 0)


def reference_03() -> int:
    w = [2, 3, 4]
    x = [10, 100, 1000]
    return sum((wi * xi for wi, xi in zip(w, x)), 0)


# ---------------------------------------------------------------------------
# 第 4 题：嵌套推导 —— 对应 Layer.parameters / MLP.parameters
#   [p for n in self.neurons for p in n.parameters()]
# 读法：先 for 外层，再 for 内层（和双层 for 一样）
# ---------------------------------------------------------------------------

def drill_04() -> List[int]:
    """把二维列表摊平：[[1, 2], [3], [4, 5]] → [1, 2, 3, 4, 5]"""
    rows = [[1, 2], [3], [4, 5]]
    return [item for row in rows for item in row]


def reference_04() -> List[int]:
    rows = [[1, 2], [3], [4, 5]]
    return [item for row in rows for item in row]


# ---------------------------------------------------------------------------
# 第 5 题：三元表达式 —— 对应 return act.tanh() if self.nonlin else act
# ---------------------------------------------------------------------------

def drill_05(nonlin: bool) -> str:
    """nonlin 为 True 返回 "tanh"，否则返回 "linear"。一行三元表达式。"""
    return "tanh" if nonlin else "linear"


def reference_05(nonlin: bool) -> str:
    return "tanh" if nonlin else "linear"


# ---------------------------------------------------------------------------
# 第 6 题：**kwargs 接收 —— 对应 Layer.__init__(..., **kwargs)
# ---------------------------------------------------------------------------

def drill_06(**kwargs: Any) -> dict:
    """原样返回收到的关键字参数字典。

    调用方会写：drill_06(nonlin=False, foo=1)
    你应返回 {"nonlin": False, "foo": 1}
    """
    # TODO: return kwargs
    return kwargs


def reference_06(**kwargs: Any) -> dict:
    return kwargs


# ---------------------------------------------------------------------------
# 第 7 题：**kwargs 转发 —— 对应 Neuron(nin, **kwargs)
# ---------------------------------------------------------------------------

def make_neuron(nin: int, nonlin: bool = True) -> dict:
    """假装的 Neuron 构造：只返回参数字典，方便检查。"""
    return {"nin": nin, "nonlin": nonlin}


def drill_07(nin: int, nout: int, **kwargs: Any) -> List[dict]:
    """模拟 Layer：创建 nout 个 neuron，把 **kwargs 转发给 make_neuron。

    例如 drill_07(3, 2, nonlin=False) 应得到：
      [{"nin": 3, "nonlin": False}, {"nin": 3, "nonlin": False}]
    """
    return [make_neuron(nin, **kwargs) for _ in range(nout)]


def reference_07(nin: int, nout: int, **kwargs: Any) -> List[dict]:
    return [make_neuron(nin, **kwargs) for _ in range(nout)]


# ---------------------------------------------------------------------------
# 第 8 题：带条件的推导 —— 对应 MLP 里 nonlin=(i != last)
# ---------------------------------------------------------------------------

def drill_08(nin: int, nouts: List[int]) -> List[tuple]:
    """模拟 MLP 建层：返回 [(in, out, nonlin), ...]

    例：drill_08(3, [4, 4, 1]) →
      [(3, 4, True), (4, 4, True), (4, 1, False)]
    最后一层 nonlin=False，其余 True。
    """
    sizes = [nin] + nouts
    # TODO: 用列表推导，参考 nn.py MLP.__init__
    return [
        (sizes[i], sizes[i + 1], i != len(nouts) - 1)
        for i in range(len(nouts))
    ]


def reference_08(nin: int, nouts: List[int]) -> List[tuple]:
    sizes = [nin] + nouts
    return [
        (sizes[i], sizes[i + 1], i != len(nouts) - 1)
        for i in range(len(nouts))
    ]


# ===========================================================================
# runner
# ===========================================================================

def _check(name: str, got: Any, expected: Any) -> bool:
    ok = got == expected
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    if not ok:
        print(f"         got     : {got!r}")
        print(f"         expected: {expected!r}")
    return ok


DRILLS: List[tuple[str, Callable[..., Any], Callable[..., Any], tuple, dict]] = [
    ("01 列表推导", drill_01, reference_01, (), {}),
    ("02 zip", drill_02, reference_02, (), {}),
    ("03 sum+生成器", drill_03, reference_03, (), {}),
    ("04 嵌套摊平", drill_04, reference_04, (), {}),
    ("05 三元", lambda: drill_05(True), lambda: reference_05(True), (), {}),
    ("05b 三元", lambda: drill_05(False), lambda: reference_05(False), (), {}),
    ("06 **kwargs 接收", lambda: drill_06(nonlin=False, foo=1), lambda: reference_06(nonlin=False, foo=1), (), {}),
    ("07 **kwargs 转发", lambda: drill_07(3, 2, nonlin=False), lambda: reference_07(3, 2, nonlin=False), (), {}),
    ("08 MLP 层表", lambda: drill_08(3, [4, 4, 1]), lambda: reference_08(3, [4, 4, 1]), (), {}),
]


def _match_drill(name: str, only: str) -> bool:
    """让 `1` / `01` / `1b` 都能选中对应题。"""
    if name.startswith(only):
        return True
    # 纯数字：1 → 匹配 01；忽略前导 0
    if only.isdigit():
        m = re.match(r"^(\d+)", name)
        return bool(m) and int(m.group(1)) == int(only)
    return False


def main(argv: List[str]) -> int:
    only = argv[1] if len(argv) > 1 else None
    print("python_drills — 填完 TODO 后重跑本文件\n")
    passed = failed = skipped = 0
    for name, fn, ref, args, kwargs in DRILLS:
        if only and not _match_drill(name, only):
            continue
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            print(f"  [SKIP] {name}  (还没写)")
            skipped += 1
            continue
        except Exception as e:
            print(f"  [FAIL] {name}  异常: {e}")
            failed += 1
            continue
        if _check(name, got, ref(*args, **kwargs)):
            passed += 1
        else:
            failed += 1
    print(f"\n通过 {passed} / 失败 {failed} / 未做 {skipped}")
    return 0 if failed == 0 and skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
