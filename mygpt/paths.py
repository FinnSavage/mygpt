"""Where datasets and trained runs live.

They always go in the project folder (next to install.py), not in whatever
folder the terminal happens to be in, so the global `mygpt` command works
from anywhere without scattering files around. Set MYGPT_HOME to use a
different folder.
"""

import os
from pathlib import Path


def _home() -> Path:
    if os.environ.get("MYGPT_HOME"):
        return Path(os.environ["MYGPT_HOME"]).expanduser().resolve()
    project = Path(__file__).resolve().parent.parent
    if (project / "pyproject.toml").exists():  # normal case: installed with `pip install -e .`
        return project
    return Path.home() / "mygpt"               # installed some other way


HOME = _home()
DATA_DIR = HOME / "data"
RUNS_DIR = HOME / "runs"
