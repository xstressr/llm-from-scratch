"""makemore 第 5 讲：WaveNet 风格层次融合。

视频的实验叙事是「三步对照」，不是「一上来就层次融合」：

    arch    结构                              参数量   视频 val
    flat3   context=3，一次压扁               ~12k     ≈2.10
    flat8   context=8，一次压扁               ~22k     ≈2.02   ← 提升主要来自这一步
    hier    context=8，三级两两融合（配平参数）~22k     ≈2.03   ← 与 flat8 基本持平
    big     hier + n_embd=24 / n_hidden=128   ~76k     ≈1.993  ← 破 2.0 靠放大容量

结论：**拉长 context 是主要功劳；参数量配平时层次结构本身在这个规模上是平手**，
再往下要靠加容量。跑 `--arch flat8` 和 `--arch hier` 对比才能看出来。

运行（在 02-makemore/ 下）：
    python wavenet.py --arch hier                  # 默认，20000 步
    python wavenet.py --arch flat8 --steps 200000  # 视频同款训练量
    python wavenet.py --bn-diagnose                # 只看 BN 在 3D 上的统计维（视频的 bug 现场）
    python wavenet.py --conv-demo                  # 只看「卷积 = 把 for 循环塞进 kernel」
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from batchnorm import BatchNorm1d, Linear, Tanh, set_training
from bigram import DATA_PATH, load_words
from mlp import (
    BATCH_SIZE,
    LR_END,
    LR_START,
    MAX_STEPS,
    SEED,
    build_dataset,
    build_vocab,
    split_words,
)


@dataclass(frozen=True)
class Config:
    """视频里依次跑过的四个配置。"""

    block_size: int
    n_embd: int
    n_hidden: int
    hierarchical: bool
    video_val: str
    note: str


CONFIGS: dict[str, Config] = {
    "flat3": Config(3, 10, 200, False, "≈2.10", "第 2-3 讲的起点"),
    "flat8": Config(8, 10, 200, False, "≈2.02", "只把 context 拉长，结构不变"),
    "hier": Config(8, 10, 68, True, "≈2.03", "层次融合，参数量与 flat8 配平"),
    "big": Config(8, 24, 128, True, "≈1.993", "层次融合 + 放大容量"),
}
DEFAULT_ARCH = "hier"


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
    """(B, T, C) → (B, T*C)。旧 MLP 的「一次压扁」。

    等价于 FlattenConsecutive(T)：一次把全部时间步吃掉。
    """

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


def build_model(cfg: Config, vocab_size: int, seed: int = SEED) -> Sequential:
    g = torch.Generator().manual_seed(seed)
    if cfg.hierarchical:
        # block_size=8 → 三次两两融合：8→4→2→1，每融合一次感受野翻倍
        layers = [
            Embedding(vocab_size, cfg.n_embd, generator=g),
            *_fuse_block(cfg.n_embd * 2, cfg.n_hidden, g),
            *_fuse_block(cfg.n_hidden * 2, cfg.n_hidden, g),
            *_fuse_block(cfg.n_hidden * 2, cfg.n_hidden, g),
            Linear(cfg.n_hidden, vocab_size, bias=True, generator=g),
        ]
    else:
        layers = [
            Embedding(vocab_size, cfg.n_embd, generator=g),
            Flatten(),
            Linear(cfg.block_size * cfg.n_embd, cfg.n_hidden, bias=False, generator=g),
            BatchNorm1d(cfg.n_hidden),
            Tanh(),
            Linear(cfg.n_hidden, vocab_size, bias=True, generator=g),
        ]

    model = Sequential(layers)
    # 最后一层压小 → 初始 logits 接近均匀，init loss ≈ log(27)
    model.layers[-1].weight.data *= 0.1
    model.layers[-1].bias.data.zero_()

    for p in model.parameters():
        p.requires_grad = True
    return model


def diagnose_bn(model: Sequential, X: torch.Tensor) -> None:
    """复现视频里 BatchNorm 的 bug 现场：3D 输入时统计维选错会怎样。

    中间层输出是 (B, T, C)。若只沿 dim=0 求均值，得到 (1, T, C) —— 等于把每个时间步
    当成独立通道，维护了 T 倍的多余统计量。正确是沿 (0, 1)，得到 (1, 1, C)。
    """
    set_training(model.layers, True)
    x = X
    found = False
    for i, layer in enumerate(model.layers):
        if isinstance(layer, BatchNorm1d) and x.ndim == 3:
            found = True
            wrong = tuple(x.mean(0, keepdim=True).shape)
            right = tuple(x.mean((0, 1), keepdim=True).shape)
            print(f"  layer[{i}] BatchNorm1d 输入 {tuple(x.shape)}")
            print(f"    mean(0)      → {wrong}   ← 错：每个时间步各一套统计量（{x.shape[1]} 倍冗余）")
            print(f"    mean((0, 1)) → {right}   ← 对：每个通道一套，时间步共享")
        x = layer(x)

    if not found:
        print("  这个架构没有 3D 的 BN 输入（flat 一上来就压扁成 2D），换 --arch hier 看")
        return

    print("  实际 running_mean 形状（本仓库 reshape 成 1D；照视频写会是 (1,1,C)）：")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, BatchNorm1d):
            print(f"    layer[{i}]  {tuple(layer.running_mean.shape)}")


def conv_demo(model: Sequential, cfg: Config, stoi: dict[str, int], words: list[str]) -> None:
    """视频最后一节：卷积不改变模型，只是把「滑过每个位置」的 for 循环塞进 kernel。"""
    if not cfg.hierarchical:
        print("  卷积演示针对层次架构，换 --arch hier")
        return

    word = "deandre" if "deandre" in words else next(w for w in words if len(w) == 7)
    X, _ = build_dataset([word], stoi, block_size=cfg.block_size)
    print(f"  取名字 '{word}'（{len(word)} 个字母 → {X.shape[0]} 条独立样本）")

    # --- 1) for 循环逐个位置 == 一次批量调用 ---
    # 关键：必须 eval 模式。训练模式下 BN 用 batch 统计，同一批样本会互相影响，
    # 逐个前向和批量前向就不等价了。
    set_training(model.layers, False)
    with torch.no_grad():
        one_by_one = torch.cat([model(X[i : i + 1]) for i in range(X.shape[0])], dim=0)
        batched = model(X)
    print(f"  逐个前向 {tuple(one_by_one.shape)} vs 批量前向 {tuple(batched.shape)}  "
          f"allclose={torch.allclose(one_by_one, batched, atol=1e-6)}")
    print("  → 批量只是并行了 8 次「独立」调用，重叠的中间结果全部重算了一遍")

    # --- 2) 第一级融合 == 一个 kernel_size=2 / stride=2 的 Conv1d ---
    emb, fc, lin = model.layers[0], model.layers[1], model.layers[2]
    c, h = cfg.n_embd, cfg.n_hidden
    with torch.no_grad():
        e = emb(X)                                  # (B, T, C)
        manual = lin(fc(e))                         # (B, T/2, H)

        # Linear 的输入是 concat([c_t, c_{t+1}])，下标 = k*C + c
        # Conv1d 权重要 (H, C, 2)，即 w[h, c, k] = W[k*C + c, h]
        w = lin.weight.T.view(h, 2, c).transpose(1, 2).contiguous()
        xc = e.transpose(1, 2)                      # (B, C, T)  Conv1d 要 channel-first
        conv_stride2 = F.conv1d(xc, w, stride=2)    # (B, H, T/2)
        same = torch.allclose(manual, conv_stride2.transpose(1, 2), atol=1e-5)
    print(f"  FlattenConsecutive(2)+Linear {tuple(manual.shape)} == "
          f"Conv1d(k=2,s=2) {tuple(conv_stride2.transpose(1, 2).shape)}  allclose={same}")

    # --- 3) stride=1：一次算出「所有偏移」的融合，stride=2 只是它的子集 ---
    with torch.no_grad():
        conv_stride1 = F.conv1d(xc, w, stride=1)    # (B, H, T-1)
        subset = torch.allclose(conv_stride1[:, :, ::2], conv_stride2, atol=1e-5)
    print(f"  Conv1d(k=2,s=1) → {tuple(conv_stride1.shape)}，其 [::2] 就是 stride=2 的结果  "
          f"allclose={subset}")

    # --- 4) 整条序列一次过 ---
    padded = [0] * cfg.block_size + [stoi[ch] for ch in word]
    with torch.no_grad():
        e_full = emb(torch.tensor([padded]))        # (1, L, C)
        conv_full = F.conv1d(e_full.transpose(1, 2), w, stride=1)
    print(f"  整条补齐序列 L={len(padded)} 一次卷积 → {tuple(conv_full.shape)}，"
          f"覆盖全部 {X.shape[0]} 个位置（原本要调模型 {X.shape[0]} 次）")
    print("  这就是 WaveNet「因果膨胀卷积」的全部意义：纯效率，不改变模型")


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
) -> list[float]:
    params = model.parameters()
    g = torch.Generator().manual_seed(seed)
    lossi: list[float] = []
    report_every = max(steps // 10, 1)

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
        lossi.append(loss.item())

        if i % report_every == 0 or i == steps - 1:
            tr = evaluate_loss(Xtr, Ytr, model)
            dev = evaluate_loss(Xdev, Ydev, model)
            print(
                f"  step {i:6d}  batch={loss.item():.4f}  "
                f"train={tr:.4f}  val={dev:.4f}  lr={lr}"
            )
    return lossi


def report_loss_curve(lossi: list[float], chunk: int = 1000, plot: bool = False) -> None:
    """逐步 loss 是一团噪声；每 chunk 步取均值才看得出趋势（视频同款技巧）。"""
    if len(lossi) < chunk:
        print(f"  步数 {len(lossi)} < {chunk}，跳过分段平均")
        return
    usable = len(lossi) - len(lossi) % chunk
    curve = torch.tensor(lossi[:usable]).view(-1, chunk).mean(1)
    print(f"  每 {chunk} 步平均（{curve.numel()} 段）：")
    print("   ", "  ".join(f"{v:.3f}" for v in curve.tolist()))

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(curve.tolist())
        plt.xlabel(f"每 {chunk} 步")
        plt.ylabel("mean batch loss")
        plt.title("wavenet training loss")
        plt.tight_layout()
        plt.savefig("wavenet_loss.png", dpi=120)
        print("    曲线已存 wavenet_loss.png")


@torch.no_grad()
def sample_names(
    model: Sequential,
    itos: dict[int, str],
    block_size: int,
    n: int = 20,
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
    ap = argparse.ArgumentParser(description="makemore 第 5 讲 WaveNet 风格层次融合")
    ap.add_argument(
        "--arch", choices=list(CONFIGS), default=DEFAULT_ARCH,
        help="视频跑过的四个配置；flat8 vs hier 是关键对照",
    )
    ap.add_argument("--steps", type=int, default=MAX_STEPS, help="训练步数（视频用 200000）")
    ap.add_argument("--bn-diagnose", action="store_true", help="只演示 BN 在 3D 上的统计维")
    ap.add_argument("--conv-demo", action="store_true", help="只演示卷积等价性")
    ap.add_argument("--plot", action="store_true", help="把分段平均 loss 曲线存成 png")
    args = ap.parse_args()

    cfg = CONFIGS[args.arch]
    words = load_words(DATA_PATH)
    stoi, itos, vocab_size = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)

    Xtr, Ytr = build_dataset(train_words, stoi, block_size=cfg.block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=cfg.block_size)
    Xte, Yte = build_dataset(test_words, stoi, block_size=cfg.block_size)

    model = build_model(cfg, vocab_size)
    n_param = sum(p.nelement() for p in model.parameters())
    print(
        f"arch={args.arch}（{cfg.note}）  block_size={cfg.block_size}  "
        f"n_embd={cfg.n_embd}  n_hidden={cfg.n_hidden}"
    )
    print(f"params={n_param}  视频参考 val={cfg.video_val}")

    if args.bn_diagnose:
        print("\n=== BatchNorm 在 3D 输入上的统计维 ===")
        diagnose_bn(model, Xtr[:32])
        return

    if args.conv_demo:
        print("\n=== 卷积 = 把「滑过每个位置」的 for 循环塞进 kernel ===")
        conv_demo(model, cfg, stoi, words)
        return

    # 形状走查（B=4）
    set_training(model.layers, False)
    x = torch.zeros((4, cfg.block_size), dtype=torch.long)
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
    lossi = train(Xtr, Ytr, Xdev, Ydev, model, steps=args.steps)
    report_loss_curve(lossi, plot=args.plot)

    print(f"\nfinal test NLL = {evaluate_loss(Xte, Yte, model):.4f}")
    g = torch.Generator().manual_seed(SEED)
    print("samples:", ", ".join(sample_names(model, itos, cfg.block_size, n=15, generator=g)))


if __name__ == "__main__":
    main()
