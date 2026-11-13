"""Command-line interface: `mygpt doctor|train|generate|tune|datasets|runs|ui`."""

import argparse
import re
from pathlib import Path

from . import sample
from .data import DATASETS
from .paths import RUNS_DIR
from . import train as train_mod
from .utils import doctor as run_doctor
from .utils import fmt_time


def cmd_doctor(args):
    print("Checking your setup...\n")
    for label, value, ok, hint in run_doctor():
        mark = "OK" if ok else "!!"
        print(f"[{mark}] {label:<22} {value}")
        if hint:
            print(f"      {hint}")


def parse_duration(text: str) -> float:
    """'90s', '15m', '1.5h', or a bare number of minutes -> seconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text.lower())
    if not m:
        raise argparse.ArgumentTypeError(f"'{text}' isn't a duration. Try 15m, 1h, or 90s.")
    value, unit = float(m.group(1)), m.group(2) or "m"
    return value * {"s": 1, "m": 60, "h": 3600}[unit]


def cmd_train(args):
    gen = train_mod.train(
        dataset=args.data,
        preset=args.size,
        run=args.run,
        max_steps=args.steps,
        compile=not args.no_compile,
        resume=not args.fresh,
        autotune=not args.no_autotune,
        fit_data=not args.no_fit_data,
        time_budget=args.time,
        early_stop=args.early_stop,
        patience=args.patience,
    )
    run_name = args.run
    try:
        for event in gen:
            t = event["type"]
            if "run" in event:
                run_name = event["run"]
            if t == "info":
                print(event["msg"])
            elif t == "step":
                msg = (
                    f"step {event['step'] + 1}/{event['max_steps']}  "
                    f"loss {event['loss']:.4f}  lr {event['lr']:.2e}  "
                    f"elapsed {fmt_time(event['elapsed'])}  eta {fmt_time(event['eta'])}"
                )
                print("\r" + msg + " " * 10, end="", flush=True)
            elif t == "eval":
                print(
                    f"\n  eval at step {event['step'] + 1}: "
                    f"train loss {event['train_loss']:.4f}, val loss {event['val_loss']:.4f}"
                )
            elif t == "sample":
                print(f"\n--- sample at step {event['step'] + 1} ---\n{event['text']}\n---")
            elif t == "done":
                name = Path(event["run_dir"]).name
                print(f"\nDone. Saved to {event['run_dir']}.")
                print(f"Try: mygpt generate --run {name}")
    except KeyboardInterrupt:
        gen.close()  # forces train()'s finally block to save a checkpoint now, not whenever gc gets to it
        if run_name and (RUNS_DIR / run_name / "ckpt.pt").exists():
            print(f"\nStopped. Progress saved to {RUNS_DIR / run_name}.")
        else:
            print("\nStopped before any progress was saved.")


def cmd_generate(args):
    runs = sample.list_runs()
    run = args.run or (runs[-1] if runs else None)
    if run is None:
        print("No trained runs yet. Run 'mygpt train' first.")
        return
    model, tok, device = sample.load_run(run)

    def run_once(prompt):
        print(prompt, end="", flush=True)
        for event in sample.generate(model, tok, device, prompt, args.tokens, args.temperature, args.top_k):
            if event["type"] == "warning":
                print(f"({event['msg']})")
            else:
                print(event["text"], end="", flush=True)
        print()

    if args.prompt is not None:
        run_once(args.prompt)
        return

    print(f"Interactive generation with run '{run}'. Empty line or Ctrl+C to quit.")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        run_once(prompt)


def cmd_runs(args):
    runs = sample.list_runs()
    if not runs:
        print("No runs yet. Run 'mygpt train' first.")
        return
    for name in runs:
        print(name)


def cmd_ui(args):
    from . import ui
    ui.main()


def cmd_tune(args):
    from . import autotune
    from .config import PRESET_ORDER, auto_preset
    from .data import prepare
    from .tokenizer import CharTokenizer
    from .train import TIME_OVERHEAD, _configs
    from .utils import device_name, get_device, vram_gb

    device = get_device()
    if args.force:
        autotune.clear_cache()
    vocab_size = CharTokenizer.load(prepare(args.data) / "tokenizer.json").vocab_size

    print(f"Benchmarking on {device} ({device_name(device)}). Results are cached, so this only runs once.\n")
    print(f"{'size':<8} {'fits':<5} {'batch':<10} {'precision':<10} {'per step':<9} {'default run'}")
    cap = auto_preset(device, vram_gb(device))
    for name in PRESET_ORDER:
        if PRESET_ORDER.index(name) > PRESET_ORDER.index(cap) and not args.all:
            print(f"{name:<8} skipped, too big for this hardware (use --all to benchmark anyway)")
            continue
        model_cfg, train_cfg = _configs(name, vocab_size)
        result = autotune.tune(model_cfg, train_cfg.batch_size * train_cfg.grad_accum, device)
        if result is None:
            print(f"{name:<8} {'no':<5}")
            continue
        batch = f"{result['batch_size']} x {result['grad_accum']}"
        total = fmt_time(train_cfg.max_steps * result["step_time"] * TIME_OVERHEAD)
        print(f"{name:<8} {'yes':<5} {batch:<10} {result['dtype']:<10} {result['step_time']:<9.3f} "
              f"~{total} for {train_cfg.max_steps:,} steps")


def cmd_datasets(args):
    for name, (_, desc) in DATASETS.items():
        print(f"{name:<14} {desc}")
    print("\nOr pass any .txt file: mygpt train --data path/to/file.txt")


def build_parser():
    p = argparse.ArgumentParser(prog="mygpt", description="Train and chat with a small GPT on your own machine.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check your Python, PyTorch, and GPU setup.").set_defaults(func=cmd_doctor)

    t = sub.add_parser("train", help="Train a model.")
    t.add_argument("--data", default="names", help=f"Built-in dataset ({', '.join(DATASETS)}) or path to a .txt file.")
    t.add_argument("--size", default="auto", choices=["auto", "tiny", "small", "medium"], help="Model size.")
    t.add_argument("--steps", type=int, default=None, help="Override the preset's number of training steps.")
    t.add_argument("--run", default=None, help="Name for this run (default: <dataset>-<size>).")
    t.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    t.add_argument("--fresh", action="store_true", help="Ignore any existing checkpoint and start over.")
    t.add_argument("--time", type=parse_duration, default=None, metavar="DURATION",
                   help="Optional time budget like 15m or 1h. Picks the size and steps that finish in about that long.")
    t.add_argument("--early-stop", action="store_true",
                   help="Stop when validation loss stops improving, and keep the best model.")
    t.add_argument("--patience", type=int, default=3,
                   help="With --early-stop, how many evaluations without improvement before stopping (default 3).")
    t.add_argument("--no-autotune", action="store_true",
                   help="Skip the hardware benchmark and use the preset's batch size and default precision.")
    t.add_argument("--no-fit-data", action="store_true",
                   help="With --size auto, size the model by hardware only, even if the dataset is small.")
    t.set_defaults(func=cmd_train)

    tu = sub.add_parser("tune", help="Benchmark every model size on this computer.")
    tu.add_argument("--data", default="names", help="Dataset whose vocabulary to benchmark with.")
    tu.add_argument("--force", action="store_true", help="Ignore cached results and re-benchmark.")
    tu.add_argument("--all", action="store_true", help="Also benchmark sizes too big for this hardware (slow).")
    tu.set_defaults(func=cmd_tune)

    g = sub.add_parser("generate", help="Generate text from a trained run.")
    g.add_argument("prompt", nargs="?", default=None, help="Prompt text. Omit for an interactive loop.")
    g.add_argument("--run", default=None, help="Run to use (default: most recently trained).")
    g.add_argument("--tokens", type=int, default=300, help="How many characters to generate.")
    g.add_argument("--temperature", type=float, default=0.8, help="Higher is more random.")
    g.add_argument("--top-k", type=int, default=200, help="Sample only from the top K most likely characters.")
    g.set_defaults(func=cmd_generate)

    sub.add_parser("datasets", help="List built-in datasets.").set_defaults(func=cmd_datasets)
    sub.add_parser("runs", help="List trained runs.").set_defaults(func=cmd_runs)
    sub.add_parser("ui", help="Launch the browser UI.").set_defaults(func=cmd_ui)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
