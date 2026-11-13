"""Regression test: a torch.compile'd model must still save and load cleanly.

torch.compile wraps the model, and the wrapper's state_dict keys get an
"_orig_mod." prefix. train.py keeps a reference to the unwrapped model and
saves that instead, so checkpoints load into a plain GPT.

maybe_compile() only compiles on a GPU with Triton, which hides this bug on
CPU-only machines (including CI), so the test forces compilation. It uses
the "eager" backend, which still wraps the model the same way but doesn't
need a C++ compiler, so it runs on Windows, Mac, and Linux alike.

Run with: python -m pytest
"""

import torch

from mygpt import paths, sample, train


def test_compiled_checkpoint_loads(run_train, monkeypatch):
    monkeypatch.setattr(
        train, "maybe_compile",
        lambda model, device, x, y, enabled=True: (torch.compile(model, backend="eager"), "on (forced for test)"),
    )
    run_train(run="compiled", max_steps=3)

    ckpt = torch.load(paths.RUNS_DIR / "compiled" / "ckpt.pt", map_location="cpu", weights_only=False)
    bad = [k for k in ckpt["model"] if k.startswith("_orig_mod.")]
    assert not bad, f"checkpoint has compiled-wrapper key prefixes: {bad[:3]}"

    model, tok, device = sample.load_run("compiled")
    text = "".join(e["text"] for e in sample.generate(model, tok, device, "ka", max_new_tokens=10))
    assert len(text) == 10
