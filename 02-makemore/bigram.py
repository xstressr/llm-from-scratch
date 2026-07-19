"""makemore 第 1 讲：字符级 bigram 语言模型。

两条路做同一件事（预测下一个字符）：
1. 计数表 → 归一化成概率
2. 单层线性层（one-hot → logits）→ softmax → NLL，梯度下降学出同一张表

运行（在 02-makemore/ 下）：
    python bigram.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

DATA_PATH = Path(__file__).resolve().parent / "data" / "names.txt"


def load_words(path: Path = DATA_PATH) -> list[str]:
    words = path.read_text(encoding="utf-8").splitlines()
    words = [w.strip().lower() for w in words if w.strip()]
    return words


def build_vocab(words: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(words)))
    print("build_vocab chars:", chars)
    # `.` = 词首 / 词尾特殊符（Karpathy 视频同款）
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    print("build_vocab stoi:", stoi)
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

    vocab = [itos[i] for i in range(len(itos))]

    return vocab, stoi, itos


def build_count_table(
    words: list[str], stoi: dict[str, int], vocab_size: int
) -> torch.Tensor:
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    # +smoothing：避免零概率（model smoothing）
    P = (N.float() + smoothing)
    P /= P.sum(dim=1, keepdim=True)
    return P


@torch.no_grad()
def sample_names(
    P: torch.Tensor,
    itos: dict[int, str],
    n: int = 10,
    generator: torch.Generator | None = None,
) -> list[str]:
    names: list[str] = []
    for _ in range(n):
        ix = 0
        out: list[str] = []
        while True:
            p = P[ix]
            ix = int(torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item())
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


def average_nll(P: torch.Tensor, words: list[str], stoi: dict[str, int]) -> float:
    """负对数似然越低越好；均匀随机 baseline ≈ log(vocab_size)。"""
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            p = P[stoi[ch1], stoi[ch2]]
            log_likelihood += torch.log(p).item()
            n += 1
    return -log_likelihood / n


def build_dataset(
    words: list[str], stoi: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[int] = []
    ys: list[int] = []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def train_neural_bigram(
    xs: torch.Tensor,
    ys: torch.Tensor,
    vocab_size: int,
    steps: int = 200,
    lr: float = 50.0,
    seed: int = 2147483647,
) -> torch.Tensor:
    """单层 W: (V, V)。one-hot(x) @ W = logits；目标是学出与计数表等价的概率。"""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    for i in range(steps):
        xenc = F.one_hot(xs, num_classes=vocab_size).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        # 平均 NLL + 轻微 L2（对应 counts 里的 smoothing 直觉）
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean()
        loss = loss + 0.01 * (W**2).mean()

        W.grad = None
        loss.backward()
        W.data -= lr * W.grad

        if i % 40 == 0 or i == steps - 1:
            print(f"  step {i:3d}  loss={loss.item():.4f}")

    return W.detach()


def main() -> None:
    words = load_words()
    vocab, stoi, itos = build_vocab(words)
    V = len(vocab)
    print(f"names: {len(words)}  vocab: {V}  chars: {''.join(vocab)}")

    # --- 1) 计数 bigram ---
    N = build_count_table(words, stoi, V)
    P = counts_to_probs(N)
    nll = average_nll(P, words, stoi)
    print(f"\n[count] average NLL = {nll:.4f}  (uniform ~ {torch.log(torch.tensor(V)).item():.4f})")

    g = torch.Generator().manual_seed(2147483647)
    print("[count] samples:", ", ".join(sample_names(P, itos, n=8, generator=g)))

    # --- 2) 神经网络 bigram（应逼近 count 模型）---
    xs, ys = build_dataset(words, stoi)
    print(f"\n[neural] bigrams in dataset: {xs.nelement()}")
    print("[neural] training...")
    W = train_neural_bigram(xs, ys, V)
    P_nn = F.softmax(W, dim=1)
    nll_nn = average_nll(P_nn, words, stoi)
    print(f"[neural] average NLL = {nll_nn:.4f}")

    g = torch.Generator().manual_seed(2147483647)
    print("[neural] samples:", ", ".join(sample_names(P_nn, itos, n=8, generator=g)))


if __name__ == "__main__":
    main()
