"""Shared test setup.

Every test runs against a throwaway MYGPT_HOME and a generated text file, so
the suite never touches your real data/ or runs/ folders and never needs the
network.
"""

import os
import random
import sys
import tempfile
from pathlib import Path

# must happen before anything imports mygpt.paths
os.environ["MYGPT_HOME"] = tempfile.mkdtemp(prefix="mygpt-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    """About 40,000 characters of made-up names, one per line."""
    rng = random.Random(0)
    syllables = ["ka", "lo", "mi", "ra", "ne", "to", "sa", "vi", "el", "an", "or", "is"]
    names = ["".join(rng.choice(syllables) for _ in range(rng.randint(2, 4))) for _ in range(6000)]
    path = tmp_path_factory.mktemp("corpus") / "testnames.txt"
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_autotune_cache(tmp_path, monkeypatch):
    from mygpt import autotune
    monkeypatch.setattr(autotune, "CACHE_PATH", tmp_path / "autotune.json")


@pytest.fixture
def run_train(corpus):
    """Train on the test corpus on CPU and return the list of events."""
    from mygpt import train

    def _run(**kwargs):
        kwargs.setdefault("dataset", str(corpus))
        kwargs.setdefault("preset", "tiny")
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("autotune", False)
        return list(train.train(**kwargs))

    return _run


def messages(events) -> str:
    return "\n".join(e["msg"] for e in events if e["type"] == "info")
