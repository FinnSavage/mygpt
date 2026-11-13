"""Character-level tokenizer: every distinct character is one token.

Simple on purpose. Replacing this with a byte-level BPE tokenizer is a good
next project; keep the same encode/decode/save/load interface and nothing
else has to change.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # characters the model never saw are skipped rather than crashing
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def unknown_chars(self, text: str) -> set[str]:
        return {c for c in text if c not in self.stoi}

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"type": "char", "chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
