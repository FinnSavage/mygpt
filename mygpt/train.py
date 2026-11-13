"""Training loop as a generator of event dicts, so the CLI and the UI share one implementation."""

import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import autotune as autotune_mod
from .config import PRESET_ORDER, PRESETS, ModelConfig, TrainConfig, auto_preset, data_preset, smaller_of
from .data import Batches, prepare
from .model import GPT
from .paths import RUNS_DIR
from .tokenizer import CharTokenizer
from .utils import (autocast, enable_fast_math, fmt_time, free_memory, get_device, is_oom, maybe_compile,
                    precision, reset_precision, set_precision, vram_gb)

TIME_OVERHEAD = 1.15   # evals, samples, and checkpoints on top of pure training steps
MIN_BUDGET_STEPS = 300  # below this a model barely learns; try a smaller size instead
MIN_BUDGET_FRACTION = 0.4  # ...and a size should get at least this share of its default steps


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def _evaluate(model, batches: Batches, cfg: TrainConfig, device: str, batch_size: int):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(cfg.eval_batches):
            x, y = batches.get(split, batch_size)
            with autocast(device):
                _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / cfg.eval_batches
    model.train()
    return losses["train"], losses["val"]


@torch.no_grad()
def _sample_text(model, tok: CharTokenizer, device: str, length: int = 200) -> str:
    model.eval()
    start = "\n" if "\n" in tok.stoi else tok.chars[0]
    idx = torch.tensor([[tok.stoi[start]]], dtype=torch.long, device=device)
    ids = list(model.generate_stream(idx, length, temperature=0.8, top_k=200))
    model.train()
    return tok.decode(ids)


def _save_checkpoint(path: Path, model, optimizer, model_cfg, train_cfg, step, data_dir, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "step": step,
        "data_dir": str(data_dir),
        **extra,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def _configs(preset: str, vocab_size: int):
    model_kwargs, train_kwargs = PRESETS[preset]
    return ModelConfig(vocab_size=vocab_size, **model_kwargs), TrainConfig(**train_kwargs)


def _plan(cap, explicit, vocab_size: int, device: str, use_autotune: bool, time_budget):
    """Choose the size (and, with a time budget, the step count) for a new run.

    Starts at the largest allowed size and steps down if it doesn't fit in
    memory, or wouldn't get enough steps in the time budget.
    Generator: yields info events, returns (preset, steps_or_None) or None.
    """
    if explicit:
        candidates = [explicit]
    else:
        candidates = list(reversed(PRESET_ORDER[:PRESET_ORDER.index(cap) + 1]))
    measure = use_autotune or time_budget is not None

    for i, name in enumerate(candidates):
        last = i == len(candidates) - 1
        if not measure:
            return name, None
        model_cfg, train_cfg = _configs(name, vocab_size)
        target = train_cfg.batch_size * train_cfg.grad_accum
        if not autotune_mod.is_cached(model_cfg, target, device):
            yield {"type": "info", "msg": f"Benchmarking the '{name}' size on this computer (one time only)..."}
        result = autotune_mod.tune(model_cfg, target, device)
        if result is None:
            yield {"type": "info", "msg": f"The '{name}' size doesn't fit in memory here."}
            continue
        if time_budget is None:
            return name, None
        steps = int(time_budget / (result["step_time"] * TIME_OVERHEAD))
        # a bigger model trained briefly loses to a smaller one trained properly,
        # so only take a size if it gets a decent share of its normal training
        needed = max(MIN_BUDGET_STEPS, int(train_cfg.max_steps * MIN_BUDGET_FRACTION))
        if steps < needed and not last:
            yield {"type": "info", "msg": f"'{name}' would only get {steps} steps in that time, trying a smaller size."}
            continue
        steps = max(50, min(steps, train_cfg.max_steps * 4))
        yield {"type": "info", "msg": f"Time budget {fmt_time(time_budget)}: '{name}' size, {steps:,} steps."}
        return name, steps
    return None


def train(dataset="names", preset="auto", run=None, max_steps=None, device=None, compile=True, resume=True,
          autotune=True, fit_data=True, time_budget=None, early_stop=False, patience=3):
    """Train a model, yielding progress events.

    autotune     benchmark this machine (cached) to pick batch size and precision
    fit_data     with preset="auto", don't pick a bigger model than the dataset can use
    time_budget  seconds; pick the size and step count that finish in about that long
    early_stop   stop once validation loss stops improving, and keep the best model
    """
    device = device or get_device()
    enable_fast_math(device)

    log_lines = []
    data_dir = prepare(dataset, log=log_lines.append)
    for msg in log_lines:
        yield {"type": "info", "msg": msg}
    tok = CharTokenizer.load(data_dir / "tokenizer.json")

    resuming_named_run = bool(run) and resume and (RUNS_DIR / run / "ckpt.pt").exists()
    budget_steps = None
    if not resuming_named_run:
        explicit = None if preset == "auto" else preset
        cap = None
        if explicit is None:
            cap = auto_preset(device, vram_gb(device))
            reason = "this hardware"
            if fit_data:
                train_tokens = len(np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r"))
                by_data = data_preset(train_tokens)
                if smaller_of(by_data, cap) != cap:
                    cap = by_data
                    reason = f"this dataset ({train_tokens:,} tokens)"
            yield {"type": "info", "msg": f"Auto-picked the '{cap}' size for {reason}."}
        plan = yield from _plan(cap, explicit, tok.vocab_size, device, autotune, time_budget)
        if plan is None:
            yield {"type": "info", "msg": "No model size fits in memory on this device."}
            return
        preset, budget_steps = plan

    run = run or f"{data_dir.name}-{preset}"
    run_dir = RUNS_DIR / run
    ckpt_path = run_dir / "ckpt.pt"
    best_path = run_dir / "best.pt"

    step = 0
    resumed = resume and ckpt_path.exists()
    if resumed:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg, train_cfg = ckpt["model_cfg"], ckpt["train_cfg"]
        step = ckpt["step"] + 1
        tok = CharTokenizer.load(run_dir / "tokenizer.json")
        yield {"type": "info", "msg": f"Resuming '{run}' from step {step}.", "run": run}
        if time_budget is not None:
            yield {"type": "info", "msg": "Time budgets only apply to new runs. Use --steps to train a resumed run longer."}
    else:
        ckpt = None
        model_cfg, train_cfg = _configs(preset, tok.vocab_size)
        if budget_steps is not None:
            train_cfg.max_steps = budget_steps

    if max_steps is not None:
        train_cfg.max_steps = max_steps

    if step >= train_cfg.max_steps:
        yield {
            "type": "info",
            "run": run,
            "msg": (
                f"Run '{run}' already trained to {step} steps. "
                f"Use --steps M (M > {step}) to keep going, or --fresh to start over."
            ),
        }
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "tokenizer.json", run_dir / "tokenizer.json")

    # best.pt only means something for the early-stopping run that wrote it
    best_val = float("inf")
    if early_stop and resumed and best_path.exists():
        best_val = torch.load(best_path, map_location="cpu", weights_only=False).get("val_loss", best_val)
    elif best_path.exists():
        best_path.unlink()

    batch_size = train_cfg.batch_size
    grad_accum = train_cfg.grad_accum
    reset_precision(device)  # the browser UI reuses one process; don't inherit a previous run's choice
    if autotune:
        target = batch_size * grad_accum
        if not autotune_mod.is_cached(model_cfg, target, device):
            yield {"type": "info", "msg": "Benchmarking this computer for this model size (one time only)..."}
        tuned = autotune_mod.tune(model_cfg, target, device)
        if tuned is not None:
            set_precision(device, autotune_mod.dtype_of(tuned))
            batch_size, grad_accum = tuned["batch_size"], tuned["grad_accum"]
            remaining = train_cfg.max_steps - step
            yield {"type": "info", "msg": (
                f"Autotune: batch {batch_size} x {grad_accum} accumulation, {tuned['dtype']}, "
                f"~{tuned['step_time']:.2f}s per step "
                f"(about {fmt_time(remaining * tuned['step_time'] * TIME_OVERHEAD)} total)."
            )}

    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizer(train_cfg.lr, train_cfg.weight_decay, device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt
    else:
        yield {"type": "info", "msg": f"Starting '{run}': {model.num_params():,} parameters on {device}.", "run": run}

    batches = Batches(data_dir, model_cfg.block_size, device)

    ex_x, ex_y = batches.get("train", batch_size)
    raw_model = model  # torch.compile() wraps the module; checkpoints must save/load the unwrapped one
    model, compile_status = maybe_compile(model, device, ex_x, ex_y, enabled=compile)
    yield {"type": "info", "msg": f"torch.compile: {compile_status}"}

    dtype, needs_scaler = precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    start_time = time.time()
    start_step = step
    last_good_step = step - 1
    bad_evals = 0

    try:
        while step < train_cfg.max_steps:
            lr = _lr_at(step, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            try:
                loss_accum = 0.0
                for _ in range(grad_accum):
                    x, y = batches.get("train", batch_size)
                    with autocast(device):
                        _, loss = model(x, y)
                        loss = loss / grad_accum
                    if needs_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    loss_accum += loss.item()

                if needs_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                optimizer.zero_grad(set_to_none=True)
                free_memory(device)
                if batch_size <= 1:
                    yield {"type": "info", "msg": "Out of memory even at batch size 1. Try a smaller --size."}
                    break
                batch_size = max(1, batch_size // 2)
                grad_accum *= 2
                yield {"type": "info", "msg": f"Out of memory, retrying with batch_size={batch_size}, grad_accum={grad_accum}."}
                continue

            last_good_step = step
            elapsed = time.time() - start_time
            steps_done = step - start_step + 1
            rate = steps_done / elapsed if elapsed > 0 else 0
            remaining = train_cfg.max_steps - step - 1
            eta = remaining / rate if rate > 0 else 0
            yield {
                "type": "step", "step": step, "max_steps": train_cfg.max_steps,
                "loss": loss_accum, "lr": lr, "elapsed": elapsed, "eta": eta,
            }

            is_last = step == train_cfg.max_steps - 1
            if step % train_cfg.eval_every == 0 or is_last:
                train_loss, val_loss = _evaluate(model, batches, train_cfg, device, batch_size)
                yield {"type": "eval", "step": step, "train_loss": train_loss, "val_loss": val_loss}
                _save_checkpoint(ckpt_path, raw_model, optimizer, model_cfg, train_cfg, step, data_dir)

                if early_stop:
                    if val_loss < best_val - 1e-4:
                        best_val, bad_evals = val_loss, 0
                        _save_checkpoint(best_path, raw_model, None, model_cfg, train_cfg, step, data_dir,
                                         val_loss=val_loss)
                    else:
                        bad_evals += 1
                        if bad_evals >= patience:
                            train_cfg.max_steps = step + 1  # so a plain re-run reports "already trained"
                            yield {"type": "info", "msg": (
                                f"\nStopping early: validation loss hasn't improved in {patience} checks. "
                                f"Keeping the best model (val loss {best_val:.4f})."
                            )}
                            break

            if step % train_cfg.sample_every == 0 or is_last:
                text = _sample_text(model, tok, device)
                yield {"type": "sample", "step": step, "text": text}

            step += 1
    finally:
        if last_good_step >= 0:
            _save_checkpoint(ckpt_path, raw_model, optimizer, model_cfg, train_cfg, last_good_step, data_dir)

    yield {"type": "done", "run_dir": str(run_dir)}
