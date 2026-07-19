"""makemore 第 2 讲：MLP 字符级语言模型（Bengio 风格）。

相对 bigram 的改动：
1. context 从 1 个字符变成 block_size 个（默认 3）
2. 不再 one-hot，而是可学习的 embedding 表 C
3. 拼起来的向量进隐藏层（tanh）再出 logits

运行（在 02-makemore/ 下）：
    python mlp.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from bigram import DATA_PATH, load_words

# --- 超参（Karpathy 视频同款量级，本地 CPU 也能跑）---
BLOCK_SIZE = 3          # 看前几个字符
N_EMBD = 10             # 每个字符的 embedding 维度
N_HIDDEN = 200          # 隐藏层宽度
BATCH_SIZE = 32
MAX_STEPS = 20000
LR_START = 0.1
LR_END = 0.01
SEED = 2147483647


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str], int]:
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(itos)


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """X: (N, block_size) 上文；Y: (N,) 下一字符。"""
    xs: list[list[int]] = []
    ys: list[int] = []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            xs.append(context.copy())
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs), torch.tensor(ys)


def split_words(
    words: list[str], seed: int = SEED
) -> tuple[list[str], list[str], list[str]]:
    g = torch.Generator().manual_seed(seed)
    words = words.copy()
    # 用 torch 打乱下标，避免依赖 random
    perm = torch.randperm(len(words), generator=g).tolist()
    words = [words[i] for i in perm]
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def init_params(
    vocab_size: int,
    n_embd: int = N_EMBD,
    n_hidden: int = N_HIDDEN,
    block_size: int = BLOCK_SIZE,
    seed: int = SEED,
) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g)
    b2 = torch.randn(vocab_size, generator=g)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


def forward(
    X: torch.Tensor, params: list[torch.Tensor]
) -> torch.Tensor:
    C, W1, b1, W2, b2 = params
    # X: (B, block_size) → emb: (B, block_size, n_embd) → 拼成 (B, block_size*n_embd)
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    return logits


@torch.no_grad()
def evaluate_loss(
    X: torch.Tensor, Y: torch.Tensor, params: list[torch.Tensor]
) -> float:
    logits = forward(X, params)
    return F.cross_entropy(logits, Y).item()


def train(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xdev: torch.Tensor,
    Ydev: torch.Tensor,
    params: list[torch.Tensor],
    steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> None:
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = forward(Xb, params)
        loss = F.cross_entropy(logits, Yb)

        for p in params:
            p.grad = None
        loss.backward()

        lr = LR_START if i < steps // 2 else LR_END
        for p in params:
            p.data -= lr * p.grad

        if i % 2000 == 0 or i == steps - 1:
            tr = evaluate_loss(Xtr, Ytr, params)
            dev = evaluate_loss(Xdev, Ydev, params)
            print(f"  step {i:5d}  batch={loss.item():.4f}  train={tr:.4f}  val={dev:.4f}  lr={lr}")


@torch.no_grad()
def sample_names(
    params: list[torch.Tensor],
    itos: dict[int, str],
    n: int = 20,
    block_size: int = BLOCK_SIZE,
    generator: torch.Generator | None = None,
) -> list[str]:
    names: list[str] = []
    for _ in range(n):
        out: list[str] = []
        context = [0] * block_size
        while True:
            logits = forward(torch.tensor([context]), params)
            probs = F.softmax(logits, dim=1)
            ix = int(
                torch.multinomial(
                    probs, num_samples=1, replacement=True, generator=generator
                ).item()
            )
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        names.append("".join(out))
    return names


def main() -> None:
    words = load_words(DATA_PATH)
    stoi, itos, vocab_size = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)

    Xtr, Ytr = build_dataset(train_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)
    Xte, Yte = build_dataset(test_words, stoi)

    print(
        f"names={len(words)}  vocab={vocab_size}  "
        f"block_size={BLOCK_SIZE}  n_embd={N_EMBD}  n_hidden={N_HIDDEN}"
    )
    print(f"split train/val/test words: {len(train_words)}/{len(dev_words)}/{len(test_words)}")
    print(f"examples train/val/test: {len(Xtr)}/{len(Xdev)}/{len(Xte)}")
    # 一个样本长什么样：上文 3 个 id → 下一字符
    print(f"example X[0]={Xtr[0].tolist()} → Y[0]={Ytr[0].item()} ({itos[Ytr[0].item()]})")

    params = init_params(vocab_size)
    n_param = sum(p.nelement() for p in params)
    print(f"\nparams: {n_param}")
    print("training...")
    train(Xtr, Ytr, Xdev, Ydev, params)

    print(f"\nfinal test NLL = {evaluate_loss(Xte, Yte, params):.4f}")
    print(f"(bigram 参考 ≈ 2.45；更低说明看更长 context 有用)")

    g = torch.Generator().manual_seed(SEED)
    samples = sample_names(params, itos, n=15, generator=g)
    print("samples:", ", ".join(samples))


if __name__ == "__main__":
    main()
