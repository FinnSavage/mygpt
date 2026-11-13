"""Download or read text, tokenize it once, then serve random batches from disk."""

import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .paths import DATA_DIR
from .tokenizer import CharTokenizer

DATASETS = {
    "tinystories": (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "About 19 MB of simple children's stories. Learns real sentences and plots.",
    ),
    "names": (
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
        "32,000 baby names, one per line. Trains in minutes and invents new names.",
    ),
}


def _clean_tinystories(text: str) -> str:
    # stories are separated by a literal "<|endoftext|>" marker; a blank line reads better
    text = text.replace("<|endoftext|>", "\n\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop characters too rare to learn (stray emoji, zero-width spaces)
    counts = Counter(text)
    return "".join(c for c in text if counts[c] >= 20)


def _clean_names(text: str) -> str:
    # most of the file is sorted alphabetically, so neighbouring names share
    # first letters and the model would learn that instead; shuffle (fixed seed)
    lines = [l for l in text.splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


CLEANERS = {"tinystories": _clean_tinystories, "names": _clean_names}


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or "custom"


def prepare(source: str, data_root: Path = DATA_DIR, log=print) -> Path:
    """Turn a built-in dataset name or a path to a .txt file into train/val token files.

    Returns the dataset folder. Skips work if it was already prepared.
    """
    path = Path(source)
    if source in DATASETS:
        name = source
    elif path.is_file():
        name = _slug(path.stem)
    else:
        known = ", ".join(DATASETS)
        raise ValueError(f"'{source}' is not a built-in dataset ({known}) or an existing text file.")

    out = Path(data_root) / name
    if (out / "train.bin").exists() and (out / "tokenizer.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)

    raw = out / "input.txt"
    if source in DATASETS:
        if not raw.exists():
            log(f"Downloading {name}...")
            urllib.request.urlretrieve(DATASETS[source][0], raw)
    else:
        raw.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    text = raw.read_text(encoding="utf-8")
    if source in CLEANERS:
        text = CLEANERS[source](text)
    if len(text) < 10_000:
        raise ValueError(f"That text is only {len(text):,} characters. Use at least 10,000 (more is much better).")

    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    split = int(len(ids) * 0.9)
    ids[:split].tofile(out / "train.bin")
    ids[split:].tofile(out / "val.bin")
    tok.save(out / "tokenizer.json")
    log(f"Prepared {name}: {len(text):,} characters, {tok.vocab_size} unique.")
    return out


class Batches:
    """Random training windows read straight from the memory-mapped token file.

    No worker processes, so there's none of the Windows multiprocessing trouble.
    """

    def __init__(self, data_dir: Path, block_size: int, device: str):
        self.data_dir = Path(data_dir)
        self.device = device
        self.block_size = block_size
        for split in ("train", "val"):
            n = len(self._data(split))
            if n <= block_size + 1:
                raise ValueError(
                    f"The {split} split has only {n} tokens, shorter than the model's context "
                    f"({block_size}). Use more text or a smaller model."
                )

    def _data(self, split):
        # reopen each time: cheap, and avoids a known memmap memory leak
        return np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")

    def get(self, split: str, batch_size: int):
        data = self._data(split)
        ix = np.random.randint(0, len(data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)
