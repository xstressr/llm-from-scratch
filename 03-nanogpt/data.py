"""Character-level data utilities for the Tiny Shakespeare experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        stoi = {char: index for index, char in enumerate(self.chars)}
        try:
            return [stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.chars[token_id] for token_id in token_ids)
        except IndexError as error:
            raise ValueError("token id is outside the tokenizer vocabulary") from error


@dataclass(frozen=True)
class TextData:
    train: torch.Tensor
    val: torch.Tensor
    tokenizer: CharTokenizer


def download_tiny_shakespeare(destination: Path) -> Path:
    """Download the small public corpus once and return its local path."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
        destination.write_bytes(response.read())
    return destination


def load_text_data(path: Path, train_fraction: float = 0.9) -> TextData:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    text = Path(path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(len(encoded) * train_fraction)
    if split_index < 2 or len(encoded) - split_index < 2:
        raise ValueError("text is too short for a train/validation split")
    return TextData(encoded[:split_index], encoded[split_index:], tokenizer)


def get_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= block_size:
        raise ValueError("token sequence must be longer than block_size")

    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    x = torch.stack([tokens[start : start + block_size] for start in starts])
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)
