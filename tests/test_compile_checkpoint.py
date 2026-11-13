"""Regression test: a torch.compile'd model must still save and load cleanly.

Bug: maybe_compile() wraps the model in an OptimizedModule, so
model.state_dict() gets keys prefixed with "_orig_mod.". If train.py saved
that state_dict directly, the checkpoint would fail to load (sample.load_run
builds a plain GPT and calls load_state_dict, which raises on key mismatch).
train.py now keeps a `raw_model` reference to the uncompiled module and saves
that instead.

maybe_compile() only actually calls torch.compile() on cuda/xpu with Triton
installed, so this test forces it on any device by monkeypatching train.py's
reference to it. torch.compile() itself works on cpu/mps too (via inductor's
C++ backend instead of Triton); it's only maybe_compile's gate that's
GPU-only, and that gate is exactly what hides this bug on a CPU-only machine.

Run directly: python tests/test_compile_checkpoint.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mygpt import sample, train, utils
from mygpt.paths import RUNS_DIR

RUN = "compile-regression-test"


def _force_compile(model, device, example_x, example_y, enabled=True):
    """Stand-in for utils.maybe_compile that always compiles, even without a GPU."""
    compiled = torch.compile(model)
    with utils.autocast(device):
        _, loss = compiled(example_x, example_y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return compiled, "on (forced for test)"


def main():
    shutil.rmtree(RUNS_DIR / RUN, ignore_errors=True)

    original_maybe_compile = train.maybe_compile
    train.maybe_compile = _force_compile
    try:
        list(train.train(
            dataset="names", preset="tiny", run=RUN,
            max_steps=5, device="cpu", compile=True, resume=False,
        ))
    finally:
        train.maybe_compile = original_maybe_compile

    ckpt = torch.load(RUNS_DIR / RUN / "ckpt.pt", map_location="cpu", weights_only=False)
    bad_keys = [k for k in ckpt["model"] if k.startswith("_orig_mod.")]
    assert not bad_keys, f"checkpoint has compiled-wrapper key prefixes: {bad_keys[:3]}"

    model, tok, device = sample.load_run(RUN)
    text = "".join(e["text"] for e in sample.generate(model, tok, device, "ka", max_new_tokens=20))
    assert len(text) == 20

    shutil.rmtree(RUNS_DIR / RUN, ignore_errors=True)
    print("PASS: compiled checkpoint saved without _orig_mod. prefixes, and loaded cleanly via sample.load_run")


if __name__ == "__main__":
    main()
