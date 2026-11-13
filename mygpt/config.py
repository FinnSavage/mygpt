"""Model and training settings, plus size presets."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 128   # how many tokens the model can see at once
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # width of each token's vector
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 1     # micro-batches per optimizer step
    max_steps: int = 2000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    eval_every: int = 250
    eval_batches: int = 20
    sample_every: int = 500


# (model settings, training settings) for each size
PRESETS = {
    "tiny": (
        dict(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0),
        dict(batch_size=32, max_steps=2000, lr=1e-3, min_lr=1e-4),
    ),
    "small": (
        dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1),
        dict(batch_size=32, max_steps=5000, lr=6e-4, min_lr=6e-5),
    ),
    "medium": (
        dict(block_size=512, n_layer=8, n_head=8, n_embd=512, dropout=0.1),
        dict(batch_size=32, max_steps=8000, lr=3e-4, min_lr=3e-5),
    ),
}

PRESET_HELP = {
    "tiny": "about 0.8M params, fine on CPU",
    "small": "about 11M params, needs a GPU with 4+ GB",
    "medium": "about 25M params, needs a GPU with 10+ GB",
}


PRESET_ORDER = ["tiny", "small", "medium"]


def data_preset(train_tokens: int) -> str:
    """Biggest size worth training on this much text. Bigger models on small
    datasets just memorize them and take longer for no benefit."""
    if train_tokens < 2_000_000:
        return "tiny"
    if train_tokens < 10_000_000:
        return "small"
    return "medium"


def smaller_of(a: str, b: str) -> str:
    return a if PRESET_ORDER.index(a) <= PRESET_ORDER.index(b) else b


def auto_preset(device: str, vram_gb: float | None) -> str:
    """Pick the biggest preset that should fit comfortably on this hardware."""
    if device == "cpu":
        return "tiny"
    if vram_gb is None:          # Apple Silicon or unknown: be moderate
        return "small"
    if vram_gb >= 10:
        return "medium"
    if vram_gb >= 4:
        return "small"
    return "tiny"
