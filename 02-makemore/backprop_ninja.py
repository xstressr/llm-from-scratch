"""makemore 第 4 讲：Becoming a Backprop Ninja（视频 4 个练习的逐步版）。

和 `backprop.py` 的分工：
- `backprop.py`   —— 只有融合版公式（视频 Exercise 2 + 3 的结论），6 个参数对齐即止
- 本文件          —— 视频完整流程，含被跳过的 Exercise 1

四个练习：
1. Exercise 1：把 cross_entropy 和 BatchNorm 全部拆成原子运算，逐个中间量手写反传，
   对全部 26 个张量做 exact 比对（应当逐位相等）
2. Exercise 2：把 CE 那一段折叠成 dlogits = (softmax - onehot) / n
3. Exercise 3：把 BN 那一段折叠成一行 dhprebn
4. Exercise 4：只用手写梯度跑完整训练（不调 loss.backward），校准 BN 后采样

运行（在 02-makemore/ 下）：
    python backprop_ninja.py                  # 四个练习全跑（训练 20000 步）
    python backprop_ninja.py --steps 0        # 只做 1-3 的梯度校验
    python backprop_ninja.py --exercise 1     # 只跑 Exercise 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from bigram import DATA_PATH, load_words
from mlp import (
    BLOCK_SIZE,
    N_EMBD,
    BATCH_SIZE,
    SEED,
    build_vocab,
    build_dataset,
    split_words,
)

# 视频这一讲把隐藏层调小到 64（Exercise 1 要为每个中间量存梯度，小一点跑得快）
N_HIDDEN = 64
EPS = 1e-5
PARAM_NAMES = ["C", "W1", "b1", "W2", "b2", "bngain", "bnbias"]


def cmp(name: str, dt: torch.Tensor, t: torch.Tensor) -> bool:
    """视频同款三列输出：exact（逐位相等）/ approximate / maxdiff。

    原子级反传（Exercise 1）走的是和 autograd 相同的浮点运算顺序，所以应当 exact=True；
    一旦用了融合公式（Exercise 2/3），运算顺序变了，就只能 approximate=True。
    """
    ex = bool(torch.all(dt == t.grad).item())
    app = bool(torch.allclose(dt, t.grad))
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"  {name:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff:.3e}")
    return ex


@dataclass
class Fwd:
    """一次「全拆开」的前向，保留每个中间量以便逐个反传。"""

    emb: torch.Tensor
    embcat: torch.Tensor
    hprebn: torch.Tensor
    bnmeani: torch.Tensor
    bndiff: torch.Tensor
    bndiff2: torch.Tensor
    bnvar: torch.Tensor
    bnvar_inv: torch.Tensor
    bnraw: torch.Tensor
    hpreact: torch.Tensor
    h: torch.Tensor
    logits: torch.Tensor
    logit_maxes: torch.Tensor
    norm_logits: torch.Tensor
    counts: torch.Tensor
    counts_sum: torch.Tensor
    counts_sum_inv: torch.Tensor
    probs: torch.Tensor
    logprobs: torch.Tensor
    loss: torch.Tensor

    def intermediates(self) -> list[tuple[str, torch.Tensor]]:
        return [
            (f.name, getattr(self, f.name)) for f in fields(self) if f.name != "loss"
        ]

    def retain_grads(self) -> None:
        for _, t in self.intermediates():
            t.retain_grad()


def init_params(vocab_size: int, seed: int = SEED) -> list[torch.Tensor]:
    """视频同款初始化：故意不让 bngain=1 / bnbias=0，否则某些梯度会退化成 0 看不出错。"""
    g = torch.Generator().manual_seed(seed)
    fan_in = N_EMBD * BLOCK_SIZE
    C = torch.randn((vocab_size, N_EMBD), generator=g)
    W1 = torch.randn((fan_in, N_HIDDEN), generator=g) * (5 / 3) / fan_in**0.5
    # b1 有 BN 时其实是冗余的（平移由 bnbias 接管），视频特意留着，好多练一个 db1
    b1 = torch.randn(N_HIDDEN, generator=g) * 0.1
    W2 = torch.randn((N_HIDDEN, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    bngain = torch.randn((1, N_HIDDEN), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, N_HIDDEN), generator=g) * 0.1

    params = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in params:
        p.requires_grad = True
    return params


def verbose_forward(
    params: list[torch.Tensor], Xb: torch.Tensor, Yb: torch.Tensor
) -> Fwd:
    """手写前向：不用 F.cross_entropy，把它拆成 7 步，BN 也拆成 6 步。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    n = Xb.shape[0]

    emb = C[Xb]                                             # (n, 3, 10) 查表
    embcat = emb.view(emb.shape[0], -1)                     # (n, 30) 拼平

    # --- Linear 1 ---
    hprebn = embcat @ W1 + b1                               # (n, 64)

    # --- BatchNorm（逐步版）---
    bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    # 注意 1/(n-1)：Bessel 校正（无偏方差），Exercise 3 的融合公式里会因此多出 n/(n-1)
    bnvar = 1 / (n - 1) * bndiff2.sum(0, keepdim=True)
    bnvar_inv = (bnvar + EPS) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # --- 非线性 ---
    h = torch.tanh(hpreact)

    # --- Linear 2 ---
    logits = h @ W2 + b2

    # --- cross entropy（逐步版，等价于 F.cross_entropy(logits, Yb)）---
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes                      # 减 max 只为数值稳定
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1                         # 写成 **-1 而非 1/x，方便逐步求导
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    return Fwd(
        emb=emb, embcat=embcat, hprebn=hprebn, bnmeani=bnmeani, bndiff=bndiff,
        bndiff2=bndiff2, bnvar=bnvar, bnvar_inv=bnvar_inv, bnraw=bnraw,
        hpreact=hpreact, h=h, logits=logits, logit_maxes=logit_maxes,
        norm_logits=norm_logits, counts=counts, counts_sum=counts_sum,
        counts_sum_inv=counts_sum_inv, probs=probs, logprobs=logprobs, loss=loss,
    )


def exercise1(
    params: list[torch.Tensor], t: Fwd, Xb: torch.Tensor, Yb: torch.Tensor
) -> None:
    """逐个中间量手写反传，全部与 autograd 做 exact 比对。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    n = Xb.shape[0]

    # --- cross entropy 段（从 loss 往回一步一步）---
    # loss = -logprobs[range(n), Yb].mean()：只有被选中的那 n 个格子有梯度，各为 -1/n
    dlogprobs = torch.zeros_like(t.logprobs)
    dlogprobs[range(n), Yb] = -1.0 / n
    # 等价写法：dlogprobs = -1.0 / n * F.one_hot(Yb, num_classes=t.logprobs.shape[1])

    dprobs = (1.0 / t.probs) * dlogprobs                    # log 的导数是 1/x
    # probs = counts * counts_sum_inv，counts_sum_inv 形状 (n,1) 被广播 → 反传要沿类别维求和
    dcounts_sum_inv = (t.counts * dprobs).sum(1, keepdim=True)
    dcounts = t.counts_sum_inv * dprobs
    dcounts_sum = (-t.counts_sum**-2) * dcounts_sum_inv
    # counts 有两条出路（→ probs 和 → counts_sum），梯度相加
    dcounts += torch.ones_like(t.counts) * dcounts_sum
    dnorm_logits = t.counts * dcounts                       # exp 的导数是它自己

    # logits 也有两条出路：直接进 norm_logits，以及经 max 进 logit_maxes
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    # max 只把梯度还给「被选中的那个下标」——这就是 one_hot 在这里的用处
    dlogits += F.one_hot(t.logits.max(1).indices, num_classes=t.logits.shape[1]) * dlogit_maxes
    # 减 max 不改变 softmax，所以这一路理论上应当完全抵消
    print(f"  (dlogit_maxes 应≈0，实际 abs().max() = {dlogit_maxes.abs().max().item():.3e})")

    # --- Linear 2 ---
    dh = dlogits @ W2.T
    dW2 = t.h.T @ dlogits
    db2 = dlogits.sum(0)

    # --- tanh ---
    dhpreact = (1.0 - t.h**2) * dh

    # --- BatchNorm（逐步版）---
    dbngain = (t.bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbndiff = t.bnvar_inv * dbnraw
    dbnvar_inv = (t.bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = (-0.5 * (t.bnvar + EPS) ** -1.5) * dbnvar_inv
    dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(t.bndiff2) * dbnvar
    # bndiff 的第二条出路（→ bndiff2 → var），补上
    dbndiff += (2 * t.bndiff) * dbndiff2
    dbnmeani = (-dbndiff).sum(0, keepdim=True)
    dhprebn = dbndiff.clone()
    # hprebn 的第二条出路（→ mean），mean 把梯度均摊回 n 个样本
    dhprebn += 1.0 / n * (torch.ones_like(t.hprebn) * dbnmeani)

    # --- Linear 1 ---
    dembcat = dhprebn @ W1.T
    dW1 = t.embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    # --- view + embedding ---
    demb = dembcat.view(t.emb.shape)
    dC = torch.zeros_like(C)
    # 视频用显式双循环强调「查表的反传就是按 id 把梯度加回去」；
    # dC.index_add_(0, Xb.view(-1), demb.view(-1, N_EMBD)) 是同一件事的快写法
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            dC[Xb[k, j]] += demb[k, j]

    checks = [
        ("logprobs", dlogprobs, t.logprobs),
        ("probs", dprobs, t.probs),
        ("counts_sum_inv", dcounts_sum_inv, t.counts_sum_inv),
        ("counts_sum", dcounts_sum, t.counts_sum),
        ("counts", dcounts, t.counts),
        ("norm_logits", dnorm_logits, t.norm_logits),
        ("logit_maxes", dlogit_maxes, t.logit_maxes),
        ("logits", dlogits, t.logits),
        ("h", dh, t.h),
        ("W2", dW2, W2),
        ("b2", db2, b2),
        ("hpreact", dhpreact, t.hpreact),
        ("bngain", dbngain, bngain),
        ("bnbias", dbnbias, bnbias),
        ("bnraw", dbnraw, t.bnraw),
        ("bnvar_inv", dbnvar_inv, t.bnvar_inv),
        ("bnvar", dbnvar, t.bnvar),
        ("bndiff2", dbndiff2, t.bndiff2),
        ("bndiff", dbndiff, t.bndiff),
        ("bnmeani", dbnmeani, t.bnmeani),
        ("hprebn", dhprebn, t.hprebn),
        ("embcat", dembcat, t.embcat),
        ("W1", dW1, W1),
        ("b1", db1, b1),
        ("emb", demb, t.emb),
        ("C", dC, C),
    ]
    n_exact = sum(cmp(name, dt, tensor) for name, dt, tensor in checks)
    print(f"  → exact: {n_exact}/{len(checks)}（原子级反传应该全部逐位相等）")


def exercise2(t: Fwd, Yb: torch.Tensor) -> None:
    """把 CE 那 7 步折叠成一行：dlogits = (softmax(logits) - onehot) / n。"""
    n = Yb.shape[0]

    # 先确认逐步前向和 F.cross_entropy 数值一致
    ref = F.cross_entropy(t.logits.detach(), Yb)
    print(f"  逐步 loss={t.loss.item():.6f}  F.cross_entropy={ref.item():.6f}  "
          f"diff={abs(t.loss.item() - ref.item()):.3e}")

    dlogits = F.softmax(t.logits.detach(), 1)
    dlogits[range(n), Yb] -= 1                              # 正确类 -1，即减 onehot
    dlogits /= n                                            # loss 取的是 mean
    cmp("logits", dlogits, t.logits)

    # 直觉自检：每行梯度之和为 0（概率此消彼长），正确类为负（要抬高该 logit）
    print(f"  每行和 abs().max() = {dlogits.sum(1).abs().max().item():.3e}（应≈0）")
    print(f"  正确类梯度最大值 = {dlogits[range(n), Yb].max().item():.3e}（应 < 0）")


def exercise3(params: list[torch.Tensor], t: Fwd, Yb: torch.Tensor) -> None:
    """把 BN 那 6 步折叠成一行 dhprebn。"""
    _, _, _, W2, _, bngain, _ = params
    n = Yb.shape[0]

    with torch.no_grad():
        dlogits = F.softmax(t.logits.detach(), 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        dh = dlogits @ W2.T
        dhpreact = (1.0 - t.h.detach() ** 2) * dh

        bnraw = t.bnraw.detach()
        # n/(n-1) 来自前向的 Bessel 校正（bnvar 用了 1/(n-1)）
        dhprebn = (
            bngain
            * t.bnvar_inv.detach()
            / n
            * (
                n * dhpreact
                - dhpreact.sum(0)
                - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
            )
        )
    cmp("hprebn", dhprebn, t.hprebn)


def manual_grads(
    params: list[torch.Tensor],
    Xb: torch.Tensor,
    Yb: torch.Tensor,
    emb: torch.Tensor,
    embcat: torch.Tensor,
    hprebn: torch.Tensor,
    bnvar_inv: torch.Tensor,
    bnraw: torch.Tensor,
    h: torch.Tensor,
    logits: torch.Tensor,
) -> list[torch.Tensor]:
    """Exercise 4 用的融合版反传，返回顺序同 PARAM_NAMES。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    n = Xb.shape[0]

    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    dhpreact = (1.0 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = (
        bngain
        * bnvar_inv
        / n
        * (
            n * dhpreact
            - dhpreact.sum(0)
            - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
        )
    )

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    dC.index_add_(0, Xb.view(-1), demb.view(-1, N_EMBD))

    return [dC, dW1, db1, dW2, db2, dbngain, dbnbias]


def exercise4(
    params: list[torch.Tensor],
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xdev: torch.Tensor,
    Ydev: torch.Tensor,
    itos: dict[int, str],
    steps: int,
) -> None:
    """只用手写梯度训练（loss.backward 全程不用），第 0 步顺便和 autograd 对一次。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    g = torch.Generator().manual_seed(SEED)
    n = BATCH_SIZE

    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # --- 前向（这里 BN 用 unbiased=True，与融合公式里的 n/(n-1) 对应）---
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + EPS) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        first = i == 0
        if first:
            for p in params:
                p.grad = None
            loss.backward()

        with torch.no_grad():
            grads = manual_grads(
                params, Xb, Yb,
                emb.detach(), embcat.detach(), hprebn.detach(),
                bnvar_inv.detach(), bnraw.detach(), h.detach(), logits.detach(),
            )
            if first:
                print("  第 0 步：手写梯度 vs autograd")
                for name, grad, p in zip(PARAM_NAMES, grads, params):
                    cmp(name, grad, p)
                print("  之后不再调 loss.backward()，全靠手写梯度更新")
                # 关掉 requires_grad：后面不需要计算图，前向能快一截
                for p in params:
                    p.requires_grad_(False)

            lr = 0.1 if i < steps // 2 else 0.01
            for p, grad in zip(params, grads):
                p.data -= lr * grad

        if i % max(steps // 10, 1) == 0 or i == steps - 1:
            print(f"  step {i:6d}/{steps}  batch loss={loss.item():.4f}  lr={lr}")

    # --- 训练完校准 BN 的全量统计（推理时没有 batch 可用）---
    with torch.no_grad():
        emb = C[Xtr]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)

    for split, X, Y in [("train", Xtr, Ytr), ("val", Xdev, Ydev)]:
        print(f"  {split} loss = {eval_loss(params, X, Y, bnmean, bnvar):.4f}")

    gs = torch.Generator().manual_seed(SEED)
    names = sample_names(params, itos, bnmean, bnvar, n=15, generator=gs)
    print("  samples:", ", ".join(names))


@torch.no_grad()
def eval_loss(
    params: list[torch.Tensor],
    X: torch.Tensor,
    Y: torch.Tensor,
    bnmean: torch.Tensor,
    bnvar: torch.Tensor,
) -> float:
    logits = infer_logits(params, X, bnmean, bnvar)
    return F.cross_entropy(logits, Y).item()


@torch.no_grad()
def infer_logits(
    params: list[torch.Tensor],
    X: torch.Tensor,
    bnmean: torch.Tensor,
    bnvar: torch.Tensor,
) -> torch.Tensor:
    C, W1, b1, W2, b2, bngain, bnbias = params
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    hpreact = bngain * (hprebn - bnmean) * (bnvar + EPS) ** -0.5 + bnbias
    h = torch.tanh(hpreact)
    return h @ W2 + b2


@torch.no_grad()
def sample_names(
    params: list[torch.Tensor],
    itos: dict[int, str],
    bnmean: torch.Tensor,
    bnvar: torch.Tensor,
    n: int = 15,
    generator: torch.Generator | None = None,
) -> list[str]:
    names: list[str] = []
    for _ in range(n):
        out: list[str] = []
        context = [0] * BLOCK_SIZE
        while True:
            logits = infer_logits(params, torch.tensor([context]), bnmean, bnvar)
            probs = F.softmax(logits, dim=1)
            ix = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        names.append("".join(out))
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="makemore 第 4 讲逐步版")
    ap.add_argument(
        "--exercise", choices=["1", "2", "3", "4", "all"], default="all",
        help="只跑某一个练习（默认全跑）",
    )
    ap.add_argument(
        "--steps", type=int, default=20000,
        help="Exercise 4 的训练步数；0 = 跳过训练（视频里用 200000）",
    )
    args = ap.parse_args()
    want = {"1", "2", "3", "4"} if args.exercise == "all" else {args.exercise}

    words = load_words(DATA_PATH)
    stoi, itos, vocab_size = build_vocab(words)
    train_words, dev_words, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)

    params = init_params(vocab_size)
    print(f"vocab={vocab_size}  n_hidden={N_HIDDEN}  batch={BATCH_SIZE}  "
          f"params={sum(p.nelement() for p in params)}")

    if want & {"1", "2", "3"}:
        g = torch.Generator().manual_seed(SEED)
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        t = verbose_forward(params, Xb, Yb)
        t.retain_grads()
        for p in params:
            p.grad = None
        t.loss.backward()
        print(f"\nloss={t.loss.item():.4f}（init 附近应接近 log27≈3.30）")

        if "1" in want:
            print("\n=== Exercise 1：逐个中间量手写反传 ===")
            exercise1(params, t, Xb, Yb)
        if "2" in want:
            print("\n=== Exercise 2：CE 折叠成 softmax - onehot ===")
            exercise2(t, Yb)
        if "3" in want:
            print("\n=== Exercise 3：BN 折叠成一行 ===")
            exercise3(params, t, Yb)

    if "4" in want and args.steps > 0:
        print(f"\n=== Exercise 4：只用手写梯度训练 {args.steps} 步 ===")
        # 重新初始化，避免沿用前面练习里被 backward 污染的 .grad
        params = init_params(vocab_size)
        exercise4(params, Xtr, Ytr, Xdev, Ydev, itos, args.steps)


if __name__ == "__main__":
    main()
