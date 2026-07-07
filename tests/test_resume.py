from __future__ import annotations

import sqlite3
from pathlib import Path

from batchr import run_batch
from batchr.cache import CacheStore
from tests import workers


def _make_files(tmp_path: Path, names: list[str]) -> list[str]:
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_text(f"content of {name}")
        paths.append(str(p))
    return paths


def test_kill_and_resume_real_worker_crash(tmp_path):
    """A worker process hard-crashes (os._exit) partway through a single-worker
    run. The batch must stop cleanly (no poisoned cache), and re-running the
    exact same fn/items/config must resume: items completed before the crash
    are served from cache without re-invoking the worker, and the crashed
    item plus everything after it completes normally."""
    names = [f"f{i}.txt" for i in range(7)] + ["crash.txt", "g8.txt", "g9.txt"]
    items = _make_files(tmp_path, names)
    cache_dir = str(tmp_path / ".batchr")

    report1 = run_batch(workers.crash_once_on_marked, items, cache_dir=cache_dir, workers=1)

    # only the items before the crash point were completed and cached
    assert report1.ok == 7
    assert {r.item for r in report1.results} == set(items[:7])
    calls_after_crash = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls_after_crash) == 7

    # no poisoned / partial cache entries
    store = CacheStore(cache_dir)
    tmp_files = list((tmp_path / ".batchr" / "objects").rglob("*.tmp"))
    assert tmp_files == []

    report2 = run_batch(workers.crash_once_on_marked, items, cache_dir=cache_dir, workers=1)

    assert report2.total == 10
    assert report2.cached == 7  # resumed from cache, no re-invocation
    assert report2.ok == 3  # crash.txt, g8.txt, g9.txt now complete
    assert report2.failed == 0

    calls_after_resume = (tmp_path / "_calls.log").read_text().splitlines()
    # exactly 3 new invocations (the previously-cached 7 were not re-run)
    assert len(calls_after_resume) == 10


def test_resume_via_simulated_partial_failure(tmp_path):
    """Fallback-style resume test: half the items fail on the first run
    (standing in for a crash), and re-running completes exactly those."""
    good = _make_files(tmp_path, [f"ok{i}.txt" for i in range(5)])
    bad = _make_files(tmp_path, [f"bad{i}.txt" for i in range(5)])
    items = good + bad
    cache_dir = str(tmp_path / ".batchr")

    report1 = run_batch(workers.fail_if_marked, items, cache_dir=cache_dir)
    assert report1.ok == 5
    assert report1.failed == 5

    # "fix" the bad items so the next run succeeds, simulating a rerun after
    # the underlying transient issue is resolved
    for path in bad:
        Path(path).rename(Path(path).with_name(Path(path).name.replace("bad", "fixed")))
    fixed_items = good + [p.replace("bad", "fixed") for p in bad]

    report2 = run_batch(workers.fail_if_marked, fixed_items, cache_dir=cache_dir)
    assert report2.cached == 5  # the good items, unchanged
    assert report2.ok == 5  # the renamed (now-good) items
    assert report2.failed == 0


def test_atomicity_no_orphan_tmp_files_and_index_matches_disk(tmp_path):
    """After both a clean run and a run with failures, no *.tmp files should
    remain, and every index row's output file must exist on disk."""
    good = _make_files(tmp_path, [f"ok{i}.txt" for i in range(4)])
    bad = _make_files(tmp_path, ["bad0.txt", "bad1.txt"])
    items = good + bad
    cache_dir = str(tmp_path / ".batchr")

    run_batch(workers.fail_if_marked, items, cache_dir=cache_dir)

    objects_dir = Path(cache_dir) / "objects"
    tmp_files = list(objects_dir.rglob("*.tmp"))
    assert tmp_files == []

    conn = sqlite3.connect(Path(cache_dir) / "index.sqlite")
    rows = conn.execute("SELECT cache_key, output_file FROM cache").fetchall()
    conn.close()

    assert len(rows) == 4  # only the 4 successful items were cached
    for _key, output_file in rows:
        # rows are stored relative to cache_dir so the cache is relocatable
        assert not Path(output_file).is_absolute()
        assert (Path(cache_dir) / output_file).exists()
