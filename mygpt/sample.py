"""Load a trained run and generate text from it, streamed one token at a time."""

from pathlib import Path

import torch

from .model import GPT
from .tokenizer import CharTokenizer
from .utils import get_device

from .paths import RUNS_DIR


def list_runs() -> list[str]:
    """Trained run names, oldest first."""
    if not RUNS_DIR.exists():
        return []
    runs = [p.name for p in RUNS_DIR.iterdir() if (p / "ckpt.pt").exists()]
    return sorted(runs, key=lambda name: (RUNS_DIR / name / "ckpt.pt").stat().st_mtime)


def load_run(run: str):
    """Returns (model, tokenizer, device) for a trained run, ready for generation."""
    run_dir = RUNS_DIR / run
    ckpt_path = run_dir / "ckpt.pt"
    if not ckpt_path.exists():
        known = ", ".join(list_runs()) or "none yet, run 'mygpt train' first"
        raise ValueError(f"No run named '{run}'. Known runs: {known}.")

    device = get_device()
    best_path = run_dir / "best.pt"  # written by early stopping; the best model, not the last one
    load_path = best_path if best_path.exists() else ckpt_path
    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    model = GPT(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.load(run_dir / "tokenizer.json")
    return model, tok, device


def generate(model, tok: CharTokenizer, device: str, prompt: str = "",
             max_new_tokens: int = 300, temperature: float = 0.8, top_k: int | None = 200):
    """Yield event dicts: a 'warning' for unknown prompt characters, then one 'token' per character."""
    unknown = tok.unknown_chars(prompt)
    if unknown:
        yield {"type": "warning", "msg": f"Ignoring characters the model never saw: {''.join(sorted(unknown))}"}

    ids = tok.encode(prompt)
    if not ids:
        start = "\n" if "\n" in tok.stoi else tok.chars[0]
        ids = [tok.stoi[start]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for next_id in model.generate_stream(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        yield {"type": "token", "text": tok.decode([next_id])}
