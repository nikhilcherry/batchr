from __future__ import annotations

import functools
import importlib
import json
import sys
import warnings
from pathlib import Path

import pytest

from batchr import run_batch
from tests import workers


def _make_files(tmp_path: Path, n: int, prefix: str = "f") -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{prefix}{i}.txt"
        p.write_text(f"{prefix} content {i}")
        paths.append(str(p))
    return paths


def test_run_batch_basic(tmp_path):
    items = _make_files(tmp_path, 3)
    report = run_batch(workers.read_upper, items, cache_dir=str(tmp_path / ".batchr"))

    assert report.total == 3
    assert report.ok == 3
    assert report.cached == 0
    assert report.failed == 0
    assert [r.status for r in report.results] == ["ok", "ok", "ok"]
    # deterministic ordering matching input order
    assert [r.item for r in report.results] == items

    import pickle

    for r in report.results:
        with open(r.output_path, "rb") as f:
            assert pickle.load(f) == Path(r.item).read_text().upper()


def test_cache_hit_second_run_calls_worker_zero_times(tmp_path):
    items = _make_files(tmp_path, 4)
    cache_dir = str(tmp_path / ".batchr")

    report1 = run_batch(workers.read_upper_counting, items, cache_dir=cache_dir)
    assert report1.ok == 4
    calls_after_first = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls_after_first) == 4

    report2 = run_batch(workers.read_upper_counting, items, cache_dir=cache_dir)
    assert report2.ok == 0
    assert report2.cached == 4
    calls_after_second = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls_after_second) == 4  # worker was not invoked again


def test_invalidation_on_file_content_change(tmp_path):
    items = _make_files(tmp_path, 3)
    cache_dir = str(tmp_path / ".batchr")

    run_batch(workers.read_upper, items, cache_dir=cache_dir)
    Path(items[1]).write_text("changed content")

    report = run_batch(workers.read_upper, items, cache_dir=cache_dir)
    assert report.ok == 1
    assert report.cached == 2
    changed = [r for r in report.results if r.status == "ok"]
    assert changed[0].item == items[1]


def test_invalidation_on_config_change(tmp_path):
    items = _make_files(tmp_path, 3)
    cache_dir = str(tmp_path / ".batchr")

    run_batch(workers.read_upper, items, cache_dir=cache_dir, config={"v": 1})
    report = run_batch(workers.read_upper, items, cache_dir=cache_dir, config={"v": 2})

    assert report.ok == 3
    assert report.cached == 0


def test_invalidation_on_function_source_change(tmp_path):
    items = _make_files(tmp_path, 3)
    cache_dir = str(tmp_path / ".batchr")

    mod_path = tmp_path / "fnmod.py"
    mod_path.write_text(
        "def process(item):\n"
        "    with open(item) as f:\n"
        "        return f.read().upper()\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        mod_v1 = importlib.import_module("fnmod")
        report1 = run_batch(mod_v1.process, items, cache_dir=cache_dir)
        assert report1.ok == 3

        mod_path.write_text(
            "def process(item):\n"
            "    with open(item) as f:\n"
            "        return f.read().upper() + '!'\n"
        )
        importlib.invalidate_caches()
        mod_v2 = importlib.reload(mod_v1)

        report2 = run_batch(mod_v2.process, items, cache_dir=cache_dir)
        assert report2.ok == 3
        assert report2.cached == 0
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("fnmod", None)


def test_failure_isolation_and_retry_only_failed(tmp_path):
    items = _make_files(tmp_path, 7)
    items += _make_files(tmp_path, 3, prefix="bad")
    cache_dir = str(tmp_path / ".batchr")

    report1 = run_batch(workers.fail_if_marked_counting, items, cache_dir=cache_dir)
    assert report1.ok == 7
    assert report1.failed == 3
    failed_items = report1.failed_items()
    assert len(failed_items) == 3
    assert all("bad" in Path(i).name for i in failed_items)
    for r in report1.results:
        if r.status == "failed":
            assert r.error is not None
            assert "ValueError" in r.error

    calls_after_first = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls_after_first) == 10

    report2 = run_batch(workers.fail_if_marked_counting, items, cache_dir=cache_dir)
    assert report2.ok == 0
    assert report2.cached == 7
    assert report2.failed == 3

    calls_after_second = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls_after_second) == 13  # only the 3 failed items ran again


def test_retries_recover_transient_failures(tmp_path):
    items = _make_files(tmp_path, 2)
    cache_dir = str(tmp_path / ".batchr")

    report = run_batch(
        workers.fail_n_times_then_succeed, items, cache_dir=cache_dir, retries=2
    )
    assert report.ok == 2
    assert report.failed == 0


def test_fail_fast_stops_scheduling_new_items(tmp_path):
    items = _make_files(tmp_path, 20)
    cache_dir = str(tmp_path / ".batchr")

    report = run_batch(
        workers.always_fail, items, cache_dir=cache_dir, workers=2, fail_fast=True
    )
    # at least one failure recorded, and not all 20 items were necessarily run
    assert report.failed >= 1
    assert report.ok == 0
    assert len(report.results) <= 20


def test_unpicklable_lambda_raises_value_error(tmp_path):
    items = _make_files(tmp_path, 1)
    with pytest.raises(ValueError):
        run_batch(lambda item: item, items, cache_dir=str(tmp_path / ".batchr"))


def test_unpicklable_closure_raises_value_error(tmp_path):
    items = _make_files(tmp_path, 1)

    def make_closure():
        def inner(item):
            return item
        return inner

    with pytest.raises(ValueError):
        run_batch(make_closure(), items, cache_dir=str(tmp_path / ".batchr"))


def test_getsource_fallback_warns_but_does_not_crash(tmp_path):
    items = _make_files(tmp_path, 2)
    fn = functools.partial(workers.read_upper_with_suffix, suffix="!")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_batch(fn, items, cache_dir=str(tmp_path / ".batchr"))

    assert report.ok == 2
    assert any("Could not retrieve source" in str(w.message) for w in caught)


def test_json_serializer(tmp_path):
    items = _make_files(tmp_path, 2)
    report = run_batch(
        workers.make_json_dict, items, cache_dir=str(tmp_path / ".batchr"),
        serializer="json",
    )
    assert report.ok == 2
    for r in report.results:
        with open(r.output_path) as f:
            data = json.load(f)
        assert data["content"] == Path(r.item).read_text().strip()


def test_force_recomputes_even_when_cached(tmp_path):
    items = _make_files(tmp_path, 3)
    cache_dir = str(tmp_path / ".batchr")

    run_batch(workers.read_upper_counting, items, cache_dir=cache_dir)
    report = run_batch(workers.read_upper_counting, items, cache_dir=cache_dir, force=True)

    assert report.ok == 3
    assert report.cached == 0
    calls = (tmp_path / "_calls.log").read_text().splitlines()
    assert len(calls) == 6


def test_report_summary_and_failed_items_format(tmp_path):
    items = _make_files(tmp_path, 2) + _make_files(tmp_path, 1, prefix="bad")
    report = run_batch(
        workers.fail_if_marked, items, cache_dir=str(tmp_path / ".batchr")
    )
    summary = report.summary()
    assert "2 ok" in summary
    assert "1 failed" in summary
    assert report.failed_items() == [i for i in items if "bad" in Path(i).name]
