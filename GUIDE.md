# The mygpt guide

A complete walkthrough for training and talking to your own GPT, written for
someone who has never done this before. If you just want the short version,
see the [README](README.md); this page explains every step.

## Contents

1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Your first training run](#3-your-first-training-run)
4. [Talking to your model](#4-talking-to-your-model)
5. [Other datasets and your own text](#5-other-datasets-and-your-own-text)
6. [Stopping, resuming, and starting over](#6-stopping-resuming-and-starting-over)
7. [Choosing a model size](#7-choosing-a-model-size)
8. [Using the browser UI](#8-using-the-browser-ui)
9. [Where everything lives on disk](#9-where-everything-lives-on-disk)
10. [Troubleshooting](#10-troubleshooting)
11. [Going further](#11-going-further)

---

## 1. What you need

- **A computer running Windows, Mac, or Linux.** A GPU speeds things up a
  lot, but isn't required - everything in this guide works on a plain CPU
  laptop, just slower.
- **Python 3.10, 3.11, 3.12, 3.13, or 3.14.** If you're not sure whether you
  have Python, or which version, open a terminal (see below) and type
  `python3 --version` (Mac/Linux) or `python --version` (Windows). If that
  fails, or shows a version outside that range, install one from
  [python.org/downloads](https://www.python.org/downloads/) and come back.
- **A terminal.** This is the text window you type commands into.
  - **Windows:** open the Start menu, type `PowerShell`, press Enter.
  - **Mac:** open Spotlight (Cmd+Space), type `Terminal`, press Enter.
  - **Linux:** you probably already know where this is.
- **About 2-3 GB of free disk space** for Python packages, more if you train
  larger models later.

You do **not** need to know anything about machine learning to follow this
guide. Where a command might be unfamiliar, this guide explains what each
part of it does.

## 2. Install

First, get the code onto your computer if you don't already have it:

```
git clone https://github.com/FinnSavage/mygpt.git
cd mygpt
```

- `git clone ...` downloads a copy of the project into a new folder called
  `mygpt`.
- `cd mygpt` moves your terminal into that folder. Every command below
  assumes you're standing inside it.

Now run the installer:

```
python install.py
```

(On some systems the command is `python3` instead of `python` - if
`python install.py` says "command not found", try `python3 install.py`.)

This one script does everything:

1. Checks your Python version is 3.10-3.14.
2. Creates a private folder called `.venv` inside the project and installs
   everything into it, so it never touches or conflicts with Python packages
   used by anything else on your computer.
3. Looks at your hardware (NVIDIA, AMD, Intel, or Apple GPU, or none) and
   installs the matching build of PyTorch, the machine-learning library this
   project is built on.
4. Installs `mygpt` itself.
5. Runs a system check and prints the result, which should look something
   like:

```
Checking your setup...

[OK] Python                 3.12.4
[OK] OS                     Darwin 24.6.0
[OK] PyTorch                2.15.0
[OK] Device                 mps (Apple Silicon GPU)
[OK] Precision              float32
[OK] Suggested model size   small
```

If every line says `[OK]`, you're ready. If something says `!!`, read the
hint it prints beneath - usually it's "no GPU found, training will be slow"
(not a problem, just slower) or a Python version issue.

This whole process can take a few minutes the first time, mostly spent
downloading PyTorch (it's a large library). You only need to run
`install.py` once per computer. If something goes wrong partway through,
it's safe to just run `python install.py` again.

### The `mygpt` command

At the end, the installer adds a `mygpt` command that works in any terminal,
from any folder, with nothing to activate. The first time, **open a new
terminal window** before using it, because terminals only learn about new
commands when they start.

(If you'd rather not have a global command, run `python install.py
--no-shortcut`. Then you activate the environment yourself in each new
terminal, with `source .venv/bin/activate` on Mac/Linux or
`.venv\Scripts\activate` on Windows.)

## 3. Your first training run

Training is the process of showing the model a lot of text, over and over,
until it gets good at predicting "what character comes next." Let's do it:

```
mygpt train
```

With no options, this:

- Downloads a built-in dataset (32,000 baby names, one per line) the first
  time you run it. After that it's cached in a
  `data/` folder and never downloaded again.
- Picks a model size that fits your hardware automatically (see
  [Choosing a model size](#7-choosing-a-model-size)). On a CPU or an
  unrecognized GPU, this is the `tiny` preset: about 0.8 million parameters,
  small enough to train in a few minutes even without a GPU.
- Starts training, printing a live-updating progress line:

```
step 142/2000  loss 2.3817  lr 9.99e-04  elapsed 22s  eta 4m 57s
```

Here's what each part means:

- **step 142/2000** - how far through training you are. The tiny preset
  trains for 2000 steps by default.
- **loss 2.3817** - a number that measures how wrong the model's
  predictions are. Lower is better. It starts around 3.3, which is
  "random guessing" among the 27 characters in this dataset (26 letters
  plus a line break), and drops as training progresses. Don't expect it to
  hit zero - names are genuinely unpredictable, so finishing around 2.0 is
  normal.
- **lr 9.99e-04** - the learning rate, how big a correction the model makes
  on each step. This rises at the start ("warmup") then gradually falls
  ("cosine decay") - you don't need to do anything about this, it's
  automatic.
- **elapsed / eta** - time spent so far, and the estimated time remaining.

Periodically you'll also see an evaluation and a text sample:

```
  eval at step 1001: train loss 2.0755, val loss 2.0993

--- sample at step 1001 ---
lumaleya
terzia
malyon
kelie
baylei
---
```

The "eval" line checks the loss on text the model trained on (`train`) versus
text it's never seen (`val`) - if these two numbers stay close together,
the model is learning general patterns rather than memorizing. The sample is
the model writing on its own, so far unsupervised: expect random letters at the very
start, then short name-shaped strings, and by the end new names that sound
plausible but mostly don't exist.

When it finishes, you'll see:

```
Done. Saved to runs/names-tiny.
Try: mygpt generate --run names-tiny
```

That's it - you've trained a language model from scratch, on your own
computer.

## 4. Talking to your model

```
mygpt generate
```

With no prompt, this starts an interactive loop using your most recently
trained run: type some starting text and press Enter, and the model
continues it, character by character, in front of you. Press Enter on an
empty line (or Ctrl+C) to quit.

**Important: this is text continuation, not a chatbot.** The model was
never taught to answer questions or follow instructions - it was only
taught to guess the next character. If you type "What is the capital of
France?" it won't answer; it'll just keep writing in whatever style it
learned, as if that question were the start of whatever it was trained on. Give it the
*beginning* of something and let it keep going - that's what it's good at.

You can also generate once and exit, without the interactive loop:

```
mygpt generate "kat" --tokens 200
```

- `"kat"` is the prompt - the text the model continues from, here the start
  of a name. (If your
  prompt has spaces, keep the quotes around it.)
- `--tokens 200` says how many characters to generate. Default is 300.

Other useful flags:

- `--run NAME` - use a specific trained run instead of the most recent one.
  See `mygpt runs` for a list of what you've trained.
- `--temperature 0.8` - how random the output is. Lower (e.g. 0.3) is more
  repetitive and "safe"; higher (e.g. 1.2) is wilder and more error-prone.
  Default is 0.8, a reasonable middle ground.
- `--top-k 200` - only consider the 200 most likely next characters at each
  step, which filters out obviously bad choices. Lower this (e.g. `--top-k
  20`) for more focused, less adventurous text.

## 5. Other datasets and your own text

### Other built-in datasets

Names are just the default. Run `mygpt datasets` to see the others:

- **`tinystories`** - about 19 MB of simple children's stories. Much bigger
  than the names list, so a `small` model learns real sentences,
  characters, and little plots. Best with a GPU.

```
mygpt train --data tinystories
mygpt generate "Once upon a time"
```

### Your own text

Instead of a built-in dataset, point `mygpt train` at any plain text file of your
own - a book, your own writing, scraped subtitles, whatever you like:

```
mygpt train --data path/to/your_file.txt
```

A few things to know:

- It needs to be a plain `.txt` file (not a `.docx`, `.pdf`, etc).
- It needs to be **at least 10,000 characters**; a lot more is much better.
  A single short story is probably too little. A full novel, or several
  years of your own journal, is a reasonable amount.
- The bigger the variety of characters in your text (emoji, accented
  letters, unusual punctuation), the bigger the model's "vocabulary" gets -
  this project tokenizes one character at a time, so every distinct
  character the text contains becomes something the model has to learn.
- The first time you use a given file, it gets processed into a `data/`
  folder, named after the file. After that it's reused automatically.
- Give the run a memorable name so it's easy to find later:

```
mygpt train --data my_journal.txt --run journal-v1
```

## 6. Stopping, resuming, and starting over

**To stop training early,** just press Ctrl+C. Progress up to that point is
saved, and you'll see a confirmation:

```
Stopped. Progress saved to runs/names-tiny.
```

**To pick up where you left off,** run the exact same command again:

```
mygpt train --run names-tiny
```

It notices the existing progress and resumes automatically - you don't lose
anything by stopping partway through. If a run has already finished all its
steps, running this again just tells you so instead of doing anything:

```
Run 'names-tiny' already trained to 2000 steps. Use --steps M (M > 2000) to keep going, or --fresh to start over.
```

**To train for longer than the preset default,** use `--steps`:

```
mygpt train --run names-tiny --steps 5000
```

**To throw away existing progress and start completely over** with the same
run name:

```
mygpt train --run names-tiny --fresh
```

**To see everything you've trained so far:**

```
mygpt runs
```

## 7. Choosing a model size

There are three sizes, each a straightforward trade: bigger models can learn
more subtle patterns, but need more memory and take longer to train.

| Size | Parameters | Needs | Good for |
|---|---|---|---|
| `tiny` | ~0.8M | Any CPU | Trying things out, no GPU, a first run |
| `small` | ~11M | GPU with 4+ GB, or patience on CPU | More convincing output, still fast to experiment with |
| `medium` | ~25M | GPU with 10+ GB | The most coherent output this project can produce |

By default, `mygpt train` uses `--size auto`, which picks the biggest size
that suits both your hardware and your dataset. A small dataset like
`names` always gets `tiny`, because a bigger model would just memorize the
list. Add `--no-fit-data` to size by hardware alone. To choose a size
yourself:

```
mygpt train --size small
```

If a size turns out to be too big for your GPU's memory, you don't need to
restart - the training loop detects the out-of-memory error, automatically
shrinks its batch size, and keeps going; you'll see a message like
`Out of memory, retrying with batch_size=16, grad_accum=2.` in the output.

### Autotune

The first time you train a given size, mygpt spends a few seconds
benchmarking your computer. It finds the largest batch that fits in memory,
tries faster number formats (like `bfloat16`) where your chip supports
them, and measures how long a step takes so it can tell you the total time
up front:

```
Autotune: batch 32 x 1 accumulation, float32, ~0.14s per step (about 5m 30s total).
```

The result is saved in `~/.cache/mygpt/`, so it only happens once per size.
To see every size at once, run `mygpt tune`. If you ever want to skip it,
add `--no-autotune`.

### Optional: a time budget

If you'd rather say how long you want to wait than pick a size, give a time
budget. mygpt picks the biggest size that can train properly in that time
and sets the number of steps to match:

```
mygpt train --data tinystories --time 30m
```

`--time` accepts things like `90s`, `15m`, or `1.5h`. It only applies to new
runs; to extend a finished run, use `--steps`.

### Optional: early stopping

With `--early-stop`, training stops once the model stops getting better on
text it hasn't seen, and `mygpt generate` uses the best version rather than
the last one. `--patience 3` (the default) is how many checks in a row
without improvement it waits before stopping.

```
mygpt train --data tinystories --early-stop
```

## 8. Using the browser UI

Everything above is also available as a point-and-click interface:

```
mygpt ui
```

This prints a local web address (something like `http://127.0.0.1:7860`) -
open it in your browser. There are three tabs:

- **Train** - pick a built-in dataset or upload your own `.txt` file, choose
  a size (the Options section has the time budget, autotune, and early
  stopping switches), hit Start, and watch the loss curve and a live sample update as it
  trains. Hit Stop at any point; like Ctrl+C in the terminal, your progress
  is saved.
- **Generate** - pick a trained run from the dropdown, type a prompt, adjust
  the sliders, and watch the model's continuation stream in.
- **System check** - the same information as `mygpt doctor`, as a table,
  with a button to re-check.

The browser tab only runs on your own computer; nothing is uploaded anywhere
unless you choose to share it yourself.

## 9. Where everything lives on disk

Nothing here is hidden away in some system folder - it's all right next to
the code, inside the `mygpt` project folder, no matter which folder you run
`mygpt` from:

- `data/<name>/` - a prepared dataset: the raw text, the tokenized version,
  and the vocabulary it learned. Safe to delete if you want to re-download
  or re-process it; it'll just get rebuilt next time it's needed.
- `runs/<name>/` - a trained model: its weights, optimizer state (so resuming
  works), and a copy of the vocabulary it was trained with. Delete a run's
  folder to permanently remove it.
- `.venv/` - the private Python environment from the installer. Safe to
  delete and recreate by running `python install.py` again, if it ever gets
  into a broken state.

None of these are tracked by git, so cloning this repo fresh on another
computer won't bring any of your trained models or datasets with it - you'd
need to copy the `runs/` and `data/` folders over yourself if you want that.

To keep datasets and runs somewhere else (a bigger drive, say), set the
`MYGPT_HOME` environment variable to that folder.

The installer also adds a small shortcut for the `mygpt` command: a link at
`~/.local/bin/mygpt` on Mac/Linux, or `mygpt.cmd` in
`%LOCALAPPDATA%\mygpt\bin` on Windows. Delete it to remove the command.

## 10. Troubleshooting

**"command not found: mygpt"** - If you just ran the installer, open a new
terminal window; the one you installed from doesn't know about the command
yet. If it still isn't found, run `python install.py` again from the
`mygpt` folder, which re-creates the command.
Every new terminal window needs this done once.

**Windows says running scripts is disabled when I activate** (only if you
installed with `--no-shortcut`) - PowerShell
blocks scripts by default. Run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer `Y`, then
activate again. This only allows scripts for your own user account.

**Typing `python` on Windows opens the Microsoft Store** - That's a
placeholder Windows ships with, not real Python. Install Python from
[python.org/downloads](https://www.python.org/downloads/) and check
"Add python.exe to PATH" during setup.

**`install.py` says my Python version isn't supported** - Install a
supported version (3.10 through 3.14) from
[python.org/downloads](https://www.python.org/downloads/), then run
`python install.py` again. You can have multiple Python versions installed
at once without conflict.

**Training seems to be using the CPU even though I have a GPU** - Run
`mygpt doctor` and check the `Device` line. If it doesn't show your GPU,
your GPU driver might need updating, or (on Windows with an older NVIDIA
card) you may need to re-run `python install.py` after updating your
driver, so it can pick a CUDA build that supports it.

**It ran out of memory and gave up** - You'll see a message like
`Out of memory even at batch size 1. Try a smaller --size.` Try a smaller
preset (`--size tiny` instead of `small`, for example), or free up memory by
closing other GPU-heavy applications.

**I trained on a tiny or weird text file and generation looks like noise** -
Likely the training text was too short or too repetitive for the model to
learn anything general. Try a bigger, more varied source text (see
[section 5](#5-other-datasets-and-your-own-text)).

**I want to delete everything and start completely fresh** - Delete the
`runs/` and `data/` folders (both just plain folders you can drag to the
trash, or `rm -rf runs data` from the terminal). Your installed environment
in `.venv/` is untouched, so you don't need to reinstall anything.

## 11. Going further

This project is intentionally small enough to read end to end. A natural
next step, once training and generating feel familiar, is to open
`mygpt/model.py` and understand - or even rewrite - the transformer itself.
Andrej Karpathy's
["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)
walks through writing almost exactly this model line by line, and is the
best starting point if you want to go from using a GPT to understanding how
one actually works. His [nanoGPT](https://github.com/karpathy/nanoGPT)
repository, which this project is directly inspired by, is the natural next
stop after that.
