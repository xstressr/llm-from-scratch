
"""makemore 第 4 讲：手写反向传播，并与 autograd 对照。

不做完整训练；只取一个 minibatch：
1. 手写 forward + backward（不用 loss.backward）
2. 同一批数据用 autograd 再算一遍
3. 逐项比对梯度是否接近

运行（在 02-makemore/ 下）：
    python backprop.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bigram import DATA_PATH, load_words
from mlp import (
    BLOCK_SIZE,
    N_EMBD,
    N_HIDDEN,
    BATCH_SIZE,
    SEED,
    build_vocab,
    build_dataset,
    split_words,
)

EPS = 1e-5


def cmp(name: str, manual: torch.Tensor, auto: torch.Tensor) -> None:
    """打印手写梯度 vs autograd 是否对齐。"""
    ok = torch.allclose(manual, auto, atol=1e-4, rtol=1e-3)
    maxdiff = (manual - auto).abs().max().item()
    print(f"  {name:12s}  allclose={ok}  max_diff={maxdiff:.2e}")


def main() -> None:
    words = load_words(DATA_PATH)
    stoi, _, vocab_size = build_vocab(words)
    train_words, _, _ = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi)

    g = torch.Generator().manual_seed(SEED)
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    B = Xb.shape[0]

    # --- 参数（与 batchnorm 讲同款尺度）---
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((vocab_size, N_EMBD), generator=g)
    W1 = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g) / (
        BLOCK_SIZE * N_EMBD
    ) ** 0.5
    # 无 b1：由 BN beta 接管
    gamma = torch.ones(N_HIDDEN)
    beta = torch.zeros(N_HIDDEN)
    W2 = torch.randn((N_HIDDEN, vocab_size), generator=g) * 0.01
    b2 = torch.zeros(vocab_size)

    params = [C, W1, gamma, beta, W2, b2]
    for p in params:
        p.requires_grad = True

    # ========== 1) autograd 路径 ==========
    emb = C[Xb]  # (B, 3, 10)
    x_in = emb.view(B, -1)  # (B, 30)
    x_lin = x_in @ W1  # (B, 200)
    bnmean = x_lin.mean(0, keepdim=True)
    bnvar = x_lin.var(0, keepdim=True, unbiased=False)
    bnstd = torch.sqrt(bnvar + EPS)
    xhat = (x_lin - bnmean) / bnstd
    hpre = gamma * xhat + beta
    h = torch.tanh(hpre)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in params:
        p.grad = None
    loss.backward()

    # 存下 autograd 梯度
    auto = {name: p.grad.clone() for name, p in zip(
        ["C", "W1", "gamma", "beta", "W2", "b2"], params
    )}

    # ========== 2) 手写反传（用同一前向中间量；断开计算图）==========
    with torch.no_grad():
        # --- CE → dlogits ---
        probs = F.softmax(logits.detach(), dim=1)
        dlogits = probs.clone()
        dlogits[range(B), Yb] -= 1
        dlogits /= B

        # --- Linear2 ---
        dh = dlogits @ W2.T
        dW2 = h.detach().T @ dlogits
        db2 = dlogits.sum(0)

        # --- tanh ---
        dhpre = dh * (1 - h.detach() ** 2)

        # --- BatchNorm ---
        dy = dhpre
        dbeta = dy.sum(0)
        dgamma = (xhat.detach() * dy).sum(0)
        dxhat = gamma.detach() * dy
        # dx_lin：紧凑公式（与 unbiased=False 的 var 一致）
        dx_lin = (
            (1.0 / B)
            * (1.0 / bnstd.detach())
            * (
                B * dxhat
                - dxhat.sum(0, keepdim=True)
                - xhat.detach() * (dxhat * xhat.detach()).sum(0, keepdim=True)
            )
        )

        # --- Linear1 ---
        dW1 = x_in.detach().T @ dx_lin
        dx_in = dx_lin @ W1.T

        # --- view + embedding ---
        demb = dx_in.view(B, BLOCK_SIZE, N_EMBD)
        dC = torch.zeros_like(C)
        dC.index_add_(0, Xb.view(-1), demb.view(-1, N_EMBD))

    manual = {
        "C": dC,
        "W1": dW1,
        "gamma": dgamma,
        "beta": dbeta,
        "W2": dW2,
        "b2": db2,
    }

    print(f"batch={B}  loss={loss.item():.4f}  (init 附近应接近 log27≈3.30)")
    print("手写梯度 vs autograd：")
    for name in ["C", "W1", "gamma", "beta", "W2", "b2"]:
        cmp(name, manual[name], auto[name])


if __name__ == "__main__":
    main()
