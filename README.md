# mygpt

*Your own GPT, trained on your own machine, on your own words.*

A small, complete, cross-platform GPT (nanoGPT-style) that trains and talks
back entirely on your own computer: no cloud, no API key, no cost, no
Linux-only tooling. Point it at a book, a script, or your own writing and
watch it learn to imitate it, one character at a time - fast enough to try
on a plain laptop CPU, faster still with a GPU, Windows, Mac, or Linux
either way.

```
python install.py
mygpt train
mygpt generate
```

`install.py` only needs to run once. It adds a `mygpt` command you can run
from any terminal, in any folder (open a new terminal window after the first
install so it picks the command up).
**New here? See [GUIDE.md](GUIDE.md) for a full walkthrough**, explained
step by step for anyone, no machine-learning background required.

## What you get

- `mygpt doctor`: check your Python, PyTorch, and GPU setup.
- `mygpt train`: train on a built-in dataset (baby names or children's
  stories; see `mygpt datasets`), your own `.txt` file,
  or resume a previous run. Benchmarks your computer once and picks the
  model size, batch size, and precision that run best on it. Optional
  `--time 15m` budget and `--early-stop`.
- `mygpt generate`: continue a prompt with a trained run, streamed to the
  terminal one character at a time. It's text continuation, not chat: give
  it the start of some text and it keeps writing.
- `mygpt tune`: show what each model size can do on this computer.
- `mygpt ui`: the same train/generate/doctor flow in a browser tab.

## How it works

Text is tokenized one character at a time (`mygpt/tokenizer.py`) and trained
with a compact GPT-2-style transformer (`mygpt/model.py`). Runs are saved to
`runs/<name>/`, datasets to `data/<name>/`, so everything lives next to the
code with no hidden state.

## Design

- **Runs anywhere PyTorch does.** NVIDIA (CUDA), AMD (ROCm), Intel (XPU),
  Apple Silicon (MPS), or plain CPU. `install.py` detects your GPU and
  installs the matching PyTorch build.
- **No Linux-only tooling.** Every entry point is Python, so nothing needs
  bash, WSL, or NCCL.
- **Single GPU.** Large effective batch sizes come from gradient
  accumulation instead of multi-GPU training.
- **Safe speedups.** Attention uses PyTorch's built-in
  `scaled_dot_product_attention`, and `torch.compile` turns on only when it
  actually works on your machine, falling back to normal mode otherwise.
- **Forgiving training.** Running out of GPU memory halves the batch size
  and keeps going, Ctrl+C saves progress, and re-running resumes where you
  left off.

## Project layout

| File | What it does |
|---|---|
| `mygpt/model.py` | The GPT model |
| `mygpt/train.py` | Training loop, checkpoints, resume |
| `mygpt/sample.py` | Loading runs and generating text |
| `mygpt/data.py` | Dataset download and batching |
| `mygpt/tokenizer.py` | Character-level tokenizer |
| `mygpt/utils.py` | Device detection and precision |
| `mygpt/cli.py` | The `mygpt` command |
| `mygpt/ui.py` | Browser UI (Gradio) |
| `install.py` | One-step installer |

## Credits

Inspired by Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and his
"Neural Networks: Zero to Hero" lectures.
