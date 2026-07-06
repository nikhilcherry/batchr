"""Module-level worker functions shared across tests.

These must live at module scope (not nested inside test functions) so that
ProcessPoolExecutor workers can import them by reference, matching how
batchr is used in real projects.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_upper(item: str) -> str:
    with open(item) as f:
        return f.read().upper()


def read_upper_counting(item: str) -> str:
    """Like read_upper, but appends a line to a sibling _calls.log on every call."""
    path = Path(item)
    counter = path.parent / "_calls.log"
    with open(counter, "a") as f:
        f.write(path.name + "\n")
    with open(path) as f:
        return f.read().upper()


def fail_if_marked(item: str) -> str:
    """Raise for any item whose filename contains 'bad'; else read_upper."""
    if "bad" in Path(item).name:
        raise ValueError(f"bad item: {item}")
    with open(item) as f:
        return f.read().upper()


def fail_if_marked_counting(item: str) -> str:
    path = Path(item)
    counter = path.parent / "_calls.log"
    with open(counter, "a") as f:
        f.write(path.name + "\n")
    if "bad" in path.name:
        raise ValueError(f"bad item: {item}")
    with open(path) as f:
        return f.read().upper()


def always_fail(item: str) -> str:
    raise RuntimeError(f"boom: {item}")


def fail_n_times_then_succeed(item: str) -> str:
    """Fails (retries - 1) times using a per-item attempt counter file, then succeeds."""
    path = Path(item)
    counter = path.parent / f"{path.name}.attempts"
    attempts = int(counter.read_text()) if counter.exists() else 0
    attempts += 1
    counter.write_text(str(attempts))
    if attempts < 3:
        raise RuntimeError(f"attempt {attempts} failed for {item}")
    with open(path) as f:
        return f.read().upper()


def crash_on_marked(item: str) -> str:
    """Hard-crashes the worker process (os._exit) for any item whose name contains 'crash'."""
    path = Path(item)
    if "crash" in path.name:
        os._exit(1)
    counter = path.parent / "_calls.log"
    with open(counter, "a") as f:
        f.write(path.name + "\n")
    with open(path) as f:
        return f.read().upper()


def crash_once_on_marked(item: str) -> str:
    """Hard-crashes (os._exit) the FIRST time it sees an item whose name contains
    'crash', then behaves normally on every subsequent call (including a later
    call for that same item) — simulates a one-time process kill that a resumed
    run should recover from."""
    path = Path(item)
    flag = path.parent / "_crashed_once.flag"
    if "crash" in path.name and not flag.exists():
        flag.write_text("crashed")
        os._exit(1)
    counter = path.parent / "_calls.log"
    with open(counter, "a") as f:
        f.write(path.name + "\n")
    with open(path) as f:
        return f.read().upper()


def read_upper_with_suffix(item: str, suffix: str = "") -> str:
    with open(item) as f:
        return f.read().upper() + suffix


def make_json_dict(item: str) -> dict:
    with open(item) as f:
        return {"content": f.read().strip()}
