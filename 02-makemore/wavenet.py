"""makemore 第 5 讲：WaveNet 风格层次融合。

阶段：
1. Embedding / Flatten / Sequential（扁平 MLP，等价 batchnorm.py）
2. FlattenConsecutive + 树状堆叠（本文件默认训这个）

运行（在 02-makemore/ 下）：
    python wavenet.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from batchnorm import BatchNorm1d, Linear, Tanh, set_training
from bigram import DATA_PATH, load_words
from mlp import (
    BATCH_SIZE,
    LR_END,
    LR_START,
    MAX_STEPS,
    N_EMBD,
    N_HIDDEN,
    SEED,
    build_dataset,
    build_vocab,
    split_words,
)

# 层次融合：context 拉长到 8（须能被 2 整除三次：8→4→2→1）
BLOCK_SIZE = 8
N_EMBD = 10
N_HIDDEN = 68  # 视频里常用量级，参数量与浅 MLP 大致可比


class Embedding:
    """查表：idx → 向量。等价于之前的 C[X]。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight[idx]

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]


class Flatten:
    """(B, T, C) → (B, T*C)。旧 MLP 的「一次压扁」。"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    def parameters(self) -> list[torch.Tensor]:
        return []


class FlattenConsecutive:
    """只拼相邻 n 个时间步：(B, T, C) → (B, T/n, C*n)。

    T 必须能被 n 整除。若 T/n == 1，挤掉中间维 → (B, C*n)，方便接最后的 Linear。
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        if t % self.n != 0:
            raise ValueError(f"T={t} not divisible by n={self.n}")
        x = x.view(b, t // self.n, c * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)  # (B, 1, C*n) → (B, C*n)
        return x

    def parameters(self) -> list[torch.Tensor]:
        return []


class Sequential:
    """按顺序跑一层层；收集全部可训练参数。"""

    def __init__(self, layers: list) -> None:
        self.layers = layers

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]


def _fuse_block(fan_in: int, n_hidden: int, g: torch.Generator) -> list:
    """相邻 n=2 拼平 → Linear → BN → Tanh。"""
    return [
        FlattenConsecutive(2),
        Linear(fan_in, n_hidden, bias=False, generator=g),
        BatchNorm1d(n_hidden),
        Tanh(),
    ]


def build_model(
    vocab_size: int,
    n_embd: int = N_EMBD,
    n_hidden: int = N_HIDDEN,
    seed: int = SEED,
) -> Sequential:
    """树状层次融合：context=8，三次两两融合后出 logits。"""
    g = torch.Generator().manual_seed(seed)
    model = Sequential(
        [
            Embedding(vocab_size, n_embd, generator=g),
            # 8 → 4
            *_fuse_block(n_embd * 2, n_hidden, g),
            # 4 → 2
            *_fuse_block(n_hidden * 2, n_hidden, g),
            # 2 → 1（FlattenConsecutive 会 squeeze 成 (B, 2*h)）
            *_fuse_block(n_hidden * 2, n_hidden, g),
            Linear(n_hidden, vocab_size, bias=True, generator=g),
        ]
    )
    model.layers[-1].weight.data *= 0.01
    model.layers[-1].bias.data.zero_()

    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def evaluate_loss(X: torch.Tensor, Y: torch.Tensor, model: Sequential) -> float:
    set_training(model.layers, False)
    return F.cross_entropy(model(X), Y).item()


def train(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xdev: torch.Tensor,
    Ydev: torch.Tensor,
    model: Sequential,
    steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> None:
    params = model.parameters()
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        set_training(model.layers, True)
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        loss = F.cross_entropy(model(Xb), Yb)
        for p in params:
            p.grad = None
        loss.backward()

        lr = LR_START if i < steps // 2 else LR_END
        for p in params:
            p.data -= lr * p.grad

        if i % 2000 == 0 or i == steps - 1:
            tr = evaluate_loss(Xtr, Ytr, model)
            dev = evaluate_loss(Xdev, Ydev, model)
            print(
                f"  step {i:5d}  batch={loss.item():.4f}  "
                f"train={tr:.4f}  val={dev:.4f}  lr={lr}"
            )


@torch.no_grad()
def sample_names(
    model: Sequential,
    itos: dict[int, str],
    n: int = 20,
    block_size: int = BLOCK_SIZE,
    generator: torch.Generator | None = None,
) -> list[str]:
    set_training(model.layers, False)
    names: list[str] = []
    for _ in range(n):
        out: list[str] = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=-1)
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

    Xtr, Ytr = build_dataset(train_words, stoi, block_size=BLOCK_SIZE)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=BLOCK_SIZE)
    Xte, Yte = build_dataset(test_words, stoi, block_size=BLOCK_SIZE)

    model = build_model(vocab_size)
    n_param = sum(p.nelement() for p in model.parameters())
    print(
        f"names={len(words)}  vocab={vocab_size}  "
        f"block_size={BLOCK_SIZE}  n_embd={N_EMBD}  n_hidden={N_HIDDEN}"
    )
    print(f"split train/val/test: {len(train_words)}/{len(dev_words)}/{len(test_words)}")
    print(f"params: {n_param}  (层次融合 WaveNet 风格)")

    # 形状走查（B=4）
    set_training(model.layers, False)
    x = torch.zeros((4, BLOCK_SIZE), dtype=torch.long)
    print("shape walk:")
    for layer in model.layers:
        x = layer(x)
        print(f"  {type(layer).__name__:20s} → {tuple(x.shape)}")

    with torch.no_grad():
        init_loss = F.cross_entropy(model(Xtr[:1000]), Ytr[:1000]).item()
    print(
        f"init loss (1k train) ≈ {init_loss:.4f}  "
        f"(uniform log27 ≈ {torch.log(torch.tensor(27.0)).item():.4f})"
    )

    print("training...")
    train(Xtr, Ytr, Xdev, Ydev, model)

    print(f"\nfinal test NLL = {evaluate_loss(Xte, Yte, model):.4f}")
    g = torch.Generator().manual_seed(SEED)
    print("samples:", ", ".join(sample_names(model, itos, n=15, generator=g)))


if __name__ == "__main__":
    main()
