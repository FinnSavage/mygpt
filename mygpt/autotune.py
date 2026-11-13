"""Benchmark this computer once per model size and remember the result.

For a given model, autotune finds:
  - the largest micro-batch that fits in memory (instead of crashing first and
    shrinking afterwards), with gradient accumulation making up the rest so the
    effective batch size, and therefore the training results, stay the same
  - the fastest precision that works (for example bf16 vs float32 on Apple
    Silicon or CPU)
  - how long one training step takes, so we can estimate total time

Results are cached in ~/.cache/mygpt/autotune.json, keyed by machine, device,
PyTorch version, and model shape, so the benchmark only runs once.
"""

import contextlib
import json
import platform
import time
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GPT
from .utils import default_precision, device_name, enable_fast_math, free_memory, is_oom, sync

CACHE_PATH = Path.home() / ".cache" / "mygpt" / "autotune.json"
WARMUP_STEPS = 2
TIMED_STEPS = 4

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _machine_key(device: str) -> str:
    return f"{platform.node()} | {device} | {device_name(device)} | torch {torch.__version__}"


def _model_key(cfg: ModelConfig, target_batch: int) -> str:
    return f"L{cfg.n_layer}-H{cfg.n_head}-E{cfg.n_embd}-T{cfg.block_size}-V{cfg.vocab_size}-B{target_batch}"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass  # caching is a convenience; never fail training over it


def is_cached(cfg: ModelConfig, target_batch: int, device: str) -> bool:
    return _model_key(cfg, target_batch) in _load_cache().get(_machine_key(device), {})


def clear_cache():
    try:
        CACHE_PATH.unlink()
    except FileNotFoundError:
        pass


def _candidate_dtypes(device: str) -> list[torch.dtype]:
    default, _ = default_precision(device)
    if device in ("cpu", "mps"):
        # bf16 can be much faster on newer chips, or slower/unsupported on older ones. Measure.
        return [torch.float32, torch.bfloat16]
    return [default]


def _autocast(device, dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def _time_steps(cfg, device, dtype, micro, accum, steps) -> tuple[float, bool]:
    """Seconds per optimizer step, and whether the loss stayed finite."""
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(1e-4, 0.1, device)
    x = torch.randint(0, cfg.vocab_size, (micro, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro, cfg.block_size), device=device)
    try:
        def one_step():
            for _ in range(accum):
                with _autocast(device, dtype):
                    _, loss = model(x, y)
                (loss / accum).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss

        for _ in range(min(WARMUP_STEPS, steps)):
            loss = one_step()
        sync(device)
        if steps <= WARMUP_STEPS:
            return 0.0, bool(torch.isfinite(loss).item())
        t0 = time.perf_counter()
        for _ in range(steps - WARMUP_STEPS):
            loss = one_step()
        sync(device)
        per_step = (time.perf_counter() - t0) / (steps - WARMUP_STEPS)
        return per_step, bool(torch.isfinite(loss).item())
    finally:
        del model, opt, x, y
        free_memory(device)


def _fits(cfg, device, dtype, micro) -> bool:
    try:
        _time_steps(cfg, device, dtype, micro, 1, 1)
        return True
    except RuntimeError as e:
        if is_oom(e):
            free_memory(device)
            return False
        raise


def tune(cfg: ModelConfig, target_batch: int, device: str, force: bool = False) -> dict | None:
    """Return {"batch_size", "grad_accum", "dtype", "step_time"} for this model on
    this machine, or None if it doesn't fit even at batch size 1."""
    enable_fast_math(device)
    cache = _load_cache()
    mkey, key = _machine_key(device), _model_key(cfg, target_batch)
    if not force and key in cache.get(mkey, {}):
        return cache[mkey][key]

    best = None
    for dtype in _candidate_dtypes(device):
        try:
            micro = target_batch
            while micro >= 1 and not _fits(cfg, device, dtype, micro):
                micro //= 2
            if micro < 1:
                continue
            accum = max(1, target_batch // micro)
            step_time, finite = _time_steps(cfg, device, dtype, micro, accum, WARMUP_STEPS + TIMED_STEPS)
        except Exception:
            continue  # this precision isn't supported here; try the next one
        if not finite:
            continue
        if best is None or step_time < best["step_time"] * 0.95:  # needs a real win to switch
            best = {
                "batch_size": micro,
                "grad_accum": accum,
                "dtype": str(dtype).replace("torch.", ""),
                "step_time": step_time,
            }

    if best is not None:
        cache.setdefault(mkey, {})[key] = best
        _save_cache(cache)
    return best


def dtype_of(result: dict) -> torch.dtype:
    return DTYPES[result["dtype"]]
