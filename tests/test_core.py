"""Fast end-to-end checks of the pieces most likely to break on a new OS."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from conftest import messages
from mygpt import autotune, cli, config, paths, sample, train
from mygpt.data import prepare
from mygpt.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


# --- data and tokenizer -------------------------------------------------------

def test_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("hello world\n")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.encode("hello?") == tok.encode("hello")  # unknown characters are skipped
    tok.save(tmp_path / "tok.json")
    assert CharTokenizer.load(tmp_path / "tok.json").chars == tok.chars


def test_prepare_writes_into_mygpt_home(corpus):
    out = prepare(str(corpus))
    assert out.parent == paths.DATA_DIR
    for name in ("train.bin", "val.bin", "tokenizer.json"):
        assert (out / name).exists()


def test_prepare_rejects_tiny_text(tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("too short", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10,000"):
        prepare(str(small))


# --- training -----------------------------------------------------------------

def test_train_resume_and_generate(run_train):
    events = run_train(run="basic", max_steps=5)
    assert events[-1]["type"] == "done"
    assert (paths.RUNS_DIR / "basic" / "ckpt.pt").exists()

    events = run_train(run="basic", max_steps=8)
    assert "Resuming 'basic' from step 5" in messages(events)
    assert [e["step"] for e in events if e["type"] == "step"] == [5, 6, 7]

    events = run_train(run="basic", max_steps=8)
    assert "already trained to 8 steps" in messages(events)

    model, tok, device = sample.load_run("basic")
    text = "".join(e["text"] for e in sample.generate(model, tok, device, "ka", max_new_tokens=20))
    assert len(text) == 20


def test_fresh_starts_over(run_train):
    run_train(run="fresh", max_steps=4)
    events = run_train(run="fresh", max_steps=3, resume=False)
    assert "Resuming" not in messages(events)
    assert [e["step"] for e in events if e["type"] == "step"] == [0, 1, 2]


def test_auto_size_respects_small_dataset(run_train):
    events = run_train(run="auto", preset="auto", max_steps=2)
    assert "Auto-picked the 'tiny' size" in messages(events)


def test_early_stop_keeps_best_model(run_train, monkeypatch):
    model_kwargs, train_kwargs = config.PRESETS["tiny"]
    monkeypatch.setitem(config.PRESETS, "tiny", (model_kwargs, {**train_kwargs, "eval_every": 2, "sample_every": 10_000}))
    fake_val = iter([3.0, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9])
    monkeypatch.setattr(train, "_evaluate", lambda *a, **k: (1.0, next(fake_val)))

    events = run_train(run="es", max_steps=100, early_stop=True, patience=2)
    assert "Stopping early" in messages(events)
    best = torch.load(paths.RUNS_DIR / "es" / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(paths.RUNS_DIR / "es" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert best["step"] == 2 and best["val_loss"] == 2.0
    assert last["step"] == 6

    # a later run without early stopping drops the stale best.pt
    monkeypatch.setattr(train, "_evaluate", lambda *a, **k: (1.0, 1.0))
    run_train(run="es", max_steps=9)
    assert not (paths.RUNS_DIR / "es" / "best.pt").exists()


# --- autotune and planning ----------------------------------------------------

def test_autotune_measures_and_caches():
    model_kwargs, _ = config.PRESETS["tiny"]
    cfg = config.ModelConfig(vocab_size=30, **model_kwargs)
    result = autotune.tune(cfg, 8, "cpu")
    assert result["batch_size"] * result["grad_accum"] == 8
    assert result["dtype"] in ("float32", "bfloat16")
    assert result["step_time"] > 0
    assert autotune.is_cached(cfg, 8, "cpu")
    assert autotune.tune(cfg, 8, "cpu") == result


def test_training_with_autotune(run_train):
    events = run_train(run="tuned", max_steps=2, autotune=True)
    assert "Autotune: batch" in messages(events)


def _drive(gen):
    msgs = []
    try:
        while True:
            msgs.append(next(gen)["msg"])
    except StopIteration as stop:
        return stop.value, msgs


def test_time_budget_prefers_a_model_that_trains_properly(monkeypatch):
    step_times = {8: 1.0, 6: 0.3, 4: 0.05}  # medium, small, tiny, keyed by n_layer
    monkeypatch.setattr(autotune, "tune", lambda cfg, *a, **k: {
        "batch_size": 32, "grad_accum": 1, "dtype": "float32", "step_time": step_times[cfg.n_layer]})
    monkeypatch.setattr(autotune, "is_cached", lambda *a: True)

    assert _drive(train._plan("medium", None, 65, "cuda", True, 600))[0][0] == "tiny"
    assert _drive(train._plan("medium", None, 65, "cuda", True, 1800))[0][0] == "small"
    assert _drive(train._plan("medium", None, 65, "cuda", True, 3 * 3600))[0][0] == "medium"


def test_plan_steps_down_when_a_size_does_not_fit(monkeypatch):
    monkeypatch.setattr(autotune, "tune", lambda cfg, *a, **k: None if cfg.n_layer == 8 else {
        "batch_size": 32, "grad_accum": 1, "dtype": "float32", "step_time": 0.1})
    monkeypatch.setattr(autotune, "is_cached", lambda *a: True)
    (preset, _), msgs = _drive(train._plan("medium", None, 65, "cuda", True, None))
    assert preset == "small"
    assert any("doesn't fit" in m for m in msgs)


def test_data_preset():
    assert config.data_preset(200_000) == "tiny"
    assert config.data_preset(5_000_000) == "small"
    assert config.data_preset(20_000_000) == "medium"


# --- CLI ----------------------------------------------------------------------

def test_parse_duration():
    assert cli.parse_duration("15m") == 900
    assert cli.parse_duration("90s") == 90
    assert cli.parse_duration("1.5h") == 5400
    assert cli.parse_duration("20") == 1200
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_duration("soon")


def test_cli_train_flags():
    args = cli.build_parser().parse_args(["train", "--time", "10m", "--early-stop", "--no-autotune"])
    assert args.time == 600 and args.early_stop and args.no_autotune and not args.no_fit_data


# --- UI and installer ---------------------------------------------------------

def test_ui_builds():
    pytest.importorskip("gradio")
    from mygpt import ui
    ui.build_demo()


def _load_installer():
    spec = importlib.util.spec_from_file_location("mygpt_install", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_python_detection():
    inst = _load_installer()
    assert inst._version_of([sys.executable]) == tuple(sys.version_info[:2])
    assert inst._supported(sys.version_info)
    assert not inst._supported((3, 9))
    assert not inst._supported(None)
    assert isinstance(inst._candidate_pythons(), list)
    assert inst._install_hint()
