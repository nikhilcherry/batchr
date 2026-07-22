"""Public API: run_batch, ItemResult, BatchReport, and the cache-key recipe.

Cache key recipe (SHA-256, updated in this exact order):
    1. the item's file content bytes if the item is a path to an existing
       file, read in chunks; else the item string itself (UTF-8)
    2. inspect.getsource(fn) — falls back to "{module}.{qualname}" with a
       logged warning if source is unavailable (e.g. lambdas, functools.partial)
    3. json.dumps(config, sort_keys=True) (empty dict if config is None)
    4. the serializer name

Any change to file contents, function source, or config therefore produces
a new key, so a change to any one of the three automatically invalidates
exactly the items it affects.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from .cache import CacheStore
from .executor import run_pool

_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ItemResult:
    item: str
    status: str  # "ok" | "failed" | "cached"
    output_path: Path | None
    error: str | None
    duration_s: float
    cache_key: str


@dataclass
class BatchReport:
    total: int
    ok: int
    cached: int
    failed: int
    results: list[ItemResult]
    wall_time_s: float

    def failed_items(self) -> list[str]:
        return [r.item for r in self.results if r.status == "failed"]

    def summary(self) -> str:
        return (
            f"{self.total} item(s): {self.ok} ok, {self.cached} cached, "
            f"{self.failed} failed, in {self.wall_time_s:.2f}s."
        )


def _fn_source(fn: Callable) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
        module = getattr(fn, "__module__", "")
        warnings.warn(
            f"Could not retrieve source for {fn!r}; falling back to "
            f"'{module}.{name}' for the cache key. Code changes to this "
            "function will NOT automatically invalidate its cache entries.",
            stacklevel=3,
        )
        return f"{module}.{name}"


def compute_cache_key(item: str, fn: Callable, config: dict | None, serializer: str) -> str:
    h = hashlib.sha256()
    path = Path(item)
    if path.is_file():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
    else:
        h.update(item.encode("utf-8"))
    h.update(_fn_source(fn).encode("utf-8"))
    h.update(json.dumps(config or {}, sort_keys=True).encode("utf-8"))
    h.update(serializer.encode("utf-8"))
    return h.hexdigest()


def _check_picklable(fn: Callable) -> None:
    try:
        pickle.dumps(fn)
    except Exception as e:
        raise ValueError(
            f"Worker function {fn!r} is not picklable ({e}). run_batch uses "
            "process-pool workers, so fn must be a module-level function "
            "(not a lambda or closure) that worker processes can import."
        ) from e


def _item_result_to_dict(r: ItemResult) -> dict:
    return {
        "item": r.item,
        "status": r.status,
        "output_path": str(r.output_path) if r.output_path is not None else None,
        "error": r.error,
        "duration_s": r.duration_s,
        "cache_key": r.cache_key,
    }


def _item_result_from_dict(d: dict) -> ItemResult:
    return ItemResult(
        item=d["item"],
        status=d["status"],
        output_path=Path(d["output_path"]) if d["output_path"] is not None else None,
        error=d["error"],
        duration_s=d["duration_s"],
        cache_key=d["cache_key"],
    )


def _report_to_dict(report: BatchReport) -> dict:
    return {
        "total": report.total,
        "ok": report.ok,
        "cached": report.cached,
        "failed": report.failed,
        "wall_time_s": report.wall_time_s,
        "results": [_item_result_to_dict(r) for r in report.results],
    }


def _report_from_dict(d: dict) -> BatchReport:
    return BatchReport(
        total=d["total"],
        ok=d["ok"],
        cached=d["cached"],
        failed=d["failed"],
        wall_time_s=d["wall_time_s"],
        results=[_item_result_from_dict(r) for r in d["results"]],
    )


def save_last_report(cache_dir: str | Path, report: BatchReport) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "last_report.json"
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_report_to_dict(report), f, indent=2)
    os.replace(tmp_path, path)


def load_last_report(cache_dir: str | Path) -> BatchReport | None:
    path = Path(cache_dir) / "last_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return _report_from_dict(json.load(f))


def run_batch(
    fn: Callable[[str], Any],
    items: Iterable[str],
    cache_dir: str | Path = ".batchr",
    config: dict | None = None,
    workers: int = 0,
    retries: int = 0,
    force: bool = False,
    fail_fast: bool = False,
    serializer: str = "pickle",
) -> BatchReport:
    if serializer not in ("pickle", "json", "npz"):
        raise ValueError(f"Unknown serializer: {serializer!r}. Must be pickle, json, or npz.")
    _check_picklable(fn)

    items = [str(item) for item in items]
    # Resolve once so the store and last_report.json agree on one directory
    # even if the process chdirs mid-run (common in notebooks).
    cache_dir = Path(cache_dir).expanduser().resolve()
    store = CacheStore(cache_dir)
    resolved_workers = workers or os.cpu_count() or 1

    start = time.monotonic()
    results: dict[int, ItemResult] = {}
    to_run: list[tuple[int, str, str]] = []
    # Two indices with the same cache key (e.g. the same item listed twice)
    # only need fn run once; every later occurrence rides on the first's result.
    primary_of_key: dict[str, int] = {}
    duplicates_of: dict[int, list[int]] = {}

    for i, item in enumerate(items):
        key = compute_cache_key(item, fn, config, serializer)
        cached_path = None if force else store.get(key)
        if cached_path is not None:
            results[i] = ItemResult(
                item=item, status="cached", output_path=cached_path,
                error=None, duration_s=0.0, cache_key=key,
            )
        elif key in primary_of_key:
            duplicates_of.setdefault(primary_of_key[key], []).append(i)
        else:
            primary_of_key[key] = i
            to_run.append((i, item, key))

    progress = tqdm(total=len(items), disable=not sys.stdout.isatty())
    counts = {"ok": 0, "cached": sum(1 for r in results.values() if r.status == "cached"), "failed": 0}
    progress.update(counts["cached"])
    progress.set_postfix(counts)

    def on_result(i: int, item: str, key: str, status: str, value: Any, error: str | None, duration: float) -> None:
        if status == "ok":
            output_path = store.put(key, item, value, serializer, getattr(fn, "__name__", ""))
            results[i] = ItemResult(
                item=item, status="ok", output_path=output_path,
                error=None, duration_s=duration, cache_key=key,
            )
            counts["ok"] += 1
        else:
            output_path = None
            results[i] = ItemResult(
                item=item, status="failed", output_path=None,
                error=error, duration_s=duration, cache_key=key,
            )
            counts["failed"] += 1
        progress.update(1)
        progress.set_postfix(counts)

        for dup_i in duplicates_of.pop(i, []):
            results[dup_i] = ItemResult(
                item=items[dup_i], status="cached" if status == "ok" else "failed",
                output_path=output_path, error=error, duration_s=0.0, cache_key=key,
            )
            counts["cached" if status == "ok" else "failed"] += 1
            progress.update(1)
            progress.set_postfix(counts)

    if to_run:
        run_pool(fn, to_run, resolved_workers, retries, fail_fast, on_result)
    progress.close()

    ordered_results = [results[i] for i in range(len(items)) if i in results]
    ok = sum(1 for r in ordered_results if r.status == "ok")
    cached = sum(1 for r in ordered_results if r.status == "cached")
    failed = sum(1 for r in ordered_results if r.status == "failed")

    report = BatchReport(
        total=len(items),
        ok=ok,
        cached=cached,
        failed=failed,
        results=ordered_results,
        wall_time_s=time.monotonic() - start,
    )
    save_last_report(cache_dir, report)
    return report
