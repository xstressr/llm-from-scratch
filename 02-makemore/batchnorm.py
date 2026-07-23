"""makemore 第 3 讲：更好的初始化 + BatchNorm。

相对 mlp.py 的改动：
1. 输出层：小 W2、零 b2 → 初始 loss ≈ log(27)
2. 隐藏层：W1 / sqrt(fan_in)；线性层无 bias（由 BN 的 β 接管）
3. Linear → BatchNorm → tanh
4. 训练用 batch 统计；推理/采样用 running 均值方差

运行（在 02-makemore/ 下）：
    python batchnorm.py
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
    MAX_STEPS,
    LR_START,
    LR_END,
    SEED,
    build_vocab,
    build_dataset,
    split_words,
)

EPS = 1e-5
BN_MOMENTUM = 0.1  # running = (1-m)*running + m*batch


class Linear:
    """y = x @ W + b（本讲隐藏层 b=None，由 BatchNorm 提供平移）。"""

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class BatchNorm1d:
    """对最后一维做 BN：训练用 batch 统计，推理用 running。"""

    def __init__(self, dim: int, eps: float = EPS, momentum: float = BN_MOMENTUM) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # 可学习
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # 推理用（不参与梯度）
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 2D (B, C)：沿 batch；3D (B, T, C)：沿 batch+时间（WaveNet 层次融合需要）
        if self.training:
            if x.ndim == 2:
                reduce_dim: int | tuple[int, ...] = 0
            elif x.ndim == 3:
                reduce_dim = (0, 1)
            else:
                raise ValueError(f"BatchNorm1d expects 2D or 3D, got ndim={x.ndim}")
            mean = x.mean(reduce_dim, keepdim=True)
            var = x.var(reduce_dim, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * mean.reshape(-1)
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * var.reshape(-1)
                )
        else:
            mean = self.running_mean
            var = self.running_var

        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def parameters(self) -> list[torch.Tensor]:
        return []


def build_model(
    vocab_size: int,
    n_embd: int = N_EMBD,
    n_hidden: int = N_HIDDEN,
    block_size: int = BLOCK_SIZE,
    seed: int = SEED,
) -> tuple[torch.Tensor, list]:
    """返回 embedding 表 C 与层列表 [Linear, BN, Tanh, Linear]。"""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    layers: list = [
        Linear(block_size * n_embd, n_hidden, bias=False, generator=g),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size, bias=True, generator=g),
    ]
    # 输出层：小权重 + 零 bias → 初始 logits 接近均匀
    layers[-1].weight.data *= 0.01
    layers[-1].bias.data.zero_()

    params = [C] + [p for layer in layers for p in layer.parameters()]
    for p in params:
        p.requires_grad = True
    return C, layers


def set_training(layers: list, training: bool) -> None:
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = training


def forward(X: torch.Tensor, C: torch.Tensor, layers: list) -> torch.Tensor:
    emb = C[X]  # (B, block_size, n_embd)
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return x  # logits


@torch.no_grad()
def evaluate_loss(
    X: torch.Tensor, Y: torch.Tensor, C: torch.Tensor, layers: list
) -> float:
    set_training(layers, False)
    logits = forward(X, C, layers)
    return F.cross_entropy(logits, Y).item()


def train(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xdev: torch.Tensor,
    Ydev: torch.Tensor,
    C: torch.Tensor,
    layers: list,
    steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> None:
    params = [C] + [p for layer in layers for p in layer.parameters()]
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        set_training(layers, True)
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = forward(Xb, C, layers)
        loss = F.cross_entropy(logits, Yb)

        for p in params:
            p.grad = None
        loss.backward()

        lr = LR_START if i < steps // 2 else LR_END
        for p in params:
            p.data -= lr * p.grad

        if i % 2000 == 0 or i == steps - 1:
            # 第一步应接近 log(27)≈3.30（若初始化正确）
            tr = evaluate_loss(Xtr, Ytr, C, layers)
            dev = evaluate_loss(Xdev, Ydev, C, layers)
            print(
                f"  step {i:5d}  batch={loss.item():.4f}  "
                f"train={tr:.4f}  val={dev:.4f}  lr={lr}"
            )


@torch.no_grad()
def sample_names(
    C: torch.Tensor,
    layers: list,
    itos: dict[int, str],
    n: int = 20,
    block_size: int = BLOCK_SIZE,
    generator: torch.Generator | None = None,
) -> list[str]:
    set_training(layers, False)
    names: list[str] = []
    for _ in range(n):
        out: list[str] = []
        context = [0] * block_size
        while True:
            logits = forward(torch.tensor([context]), C, layers)
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
    print(f"split train/val/test: {len(train_words)}/{len(dev_words)}/{len(test_words)}")

    C, layers = build_model(vocab_size)
    n_param = sum(p.nelement() for p in [C] + [p for layer in layers for p in layer.parameters()])
    print(f"params: {n_param}")

    # 初始化后立刻看一眼 loss（应接近 log 27 ≈ 3.30）
    set_training(layers, False)
    with torch.no_grad():
        init_loss = F.cross_entropy(forward(Xtr[:1000], C, layers), Ytr[:1000]).item()
    print(f"init loss (1k train) ≈ {init_loss:.4f}  (uniform log27 ≈ {torch.log(torch.tensor(27.0)).item():.4f})")

    print("training...")
    train(Xtr, Ytr, Xdev, Ydev, C, layers)

    print(f"\nfinal test NLL = {evaluate_loss(Xte, Yte, C, layers):.4f}")
    print("(mlp 无 BN 参考 ≈ 2.35)")

    g = torch.Generator().manual_seed(SEED)
    samples = sample_names(C, layers, itos, n=15, generator=g)
    print("samples:", ", ".join(samples))


if __name__ == "__main__":
    main()
