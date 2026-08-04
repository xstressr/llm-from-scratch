"""Training loop shared by the nanoGPT notebooks and command-line smoke runs."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data import TextData, download_tiny_shakespeare, get_batch, load_text_data
from model import GPT, GPTConfig


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = "data/tinyshakespeare.txt"
    run_dir: str = "runs/baseline"
    seed: int = 42
    batch_size: int = 32
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    learning_rate: float = 3e-4
    max_iters: int = 2_000
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    device: str = "auto"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: TextData,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    eval_generator = torch.Generator().manual_seed(config.seed + 1)
    for split, tokens in (("train", data.train), ("val", data.val)):
        losses = torch.zeros(config.eval_iters)
        for step in range(config.eval_iters):
            x, y = get_batch(
                tokens,
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=eval_generator,
            )
            _, loss = model(x, y)
            assert loss is not None
            losses[step] = loss.detach().cpu()
        result[split] = losses.mean().item()
    model.train()
    return result


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    tokenizer_chars: tuple[str, ...],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": asdict(config),
            "model_config": model.config.to_dict(),
            "tokenizer_chars": tokenizer_chars,
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[GPT, dict[str, Any], torch.optim.Optimizer]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    train_config = TrainConfig(**checkpoint["train_config"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, checkpoint, optimizer


def run_experiment(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_path = Path(config.data_path)
    download_tiny_shakespeare(data_path)
    data = load_text_data(data_path)

    model_config = GPTConfig(
        vocab_size=data.tokenizer.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()
    training_seconds = 0.0
    train_generator = torch.Generator().manual_seed(config.seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"device={device} parameters={model.num_parameters():,}")
    model.train()
    for step in range(config.max_iters + 1):
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss(model, data, config, device)
            elapsed = time.perf_counter() - started_at
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    step * config.batch_size * config.block_size / training_seconds
                    if training_seconds > 0
                    else 0.0
                ),
            }
            history.append(record)
            print(
                f"step={step:5d} train={losses['train']:.4f} "
                f"val={losses['val']:.4f} elapsed={elapsed:.1f}s"
            )
        if step == config.max_iters:
            break

        x, y = get_batch(
            data.train,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=train_generator,
        )
        synchronize(device)
        train_step_started_at = time.perf_counter()
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        synchronize(device)
        training_seconds += time.perf_counter() - train_step_started_at
        if step % config.log_interval == 0:
            print(f"train step={step:5d} batch_loss={loss.item():.4f}")

    checkpoint_path = run_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        data.tokenizer.chars,
        config.max_iters,
    )
    metrics = {
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameters": model.num_parameters(),
        "device": str(device),
        "training_tokens": config.max_iters * config.batch_size * config.block_size,
        "training_seconds": training_seconds,
        "tokens_per_second": (
            config.max_iters * config.batch_size * config.block_size / training_seconds
            if training_seconds > 0
            else 0.0
        ),
        "max_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**metrics, "model": model, "data": data}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny local verification")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.smoke:
        return TrainConfig(
            run_dir="runs/smoke",
            batch_size=4,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            max_iters=5,
            eval_interval=5,
            eval_iters=2,
            log_interval=1,
            device=args.device,
        )
    return TrainConfig(device=args.device)


if __name__ == "__main__":
    run_experiment(parse_args())
