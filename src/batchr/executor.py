"""Process-pool execution of the work queue.

Workers are OS processes (ProcessPoolExecutor), never threads, since the
target workload is CPU-bound scientific code and fn must be picklable.
Only the parent process talks to the cache/SQLite index; this module hands
raw (status, value, error, duration) tuples back to the caller via a
callback so the caller can commit the cache entry itself.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable


def _run_one(fn: Callable, item: str, retries: int):
    start = time.monotonic()
    last_error = None
    for _ in range(retries + 1):
        try:
            result = fn(item)
            return ("ok", result, None, time.monotonic() - start)
        except Exception:
            last_error = traceback.format_exc()
    return ("failed", None, last_error, time.monotonic() - start)


def run_pool(
    fn: Callable,
    to_run: list[tuple[int, str, str]],
    workers: int,
    retries: int,
    fail_fast: bool,
    on_result: Callable[[int, str, str, str, Any, str | None, float], None],
) -> bool:
    """Schedule ``to_run`` items across a process pool.

    ``to_run`` is a list of (index, item, cache_key) tuples. ``on_result``
    is invoked in this (parent) process for every item that finishes.

    Returns True if the run was aborted early (fail_fast triggered, or a
    worker process crashed hard and broke the pool). Items not yet started
    when that happens are simply left uncached, so a subsequent run_batch
    call will pick them up again.
    """
    aborted = False
    crashed = False

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        pending = iter(to_run)

        def submit_next() -> bool:
            try:
                i, item, key = next(pending)
            except StopIteration:
                return False
            fut = pool.submit(_run_one, fn, item, retries)
            futures[fut] = (i, item, key)
            return True

        for _ in range(workers):
            if not submit_next():
                break

        while futures:
            try:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            except BrokenProcessPool:
                crashed = True
                break

            for fut in done:
                i, item, key = futures.pop(fut)
                try:
                    status, value, error, duration = fut.result()
                except BrokenProcessPool:
                    crashed = True
                    continue
                on_result(i, item, key, status, value, error, duration)
                if status == "failed" and fail_fast:
                    aborted = True

            if crashed:
                break
            if not aborted:
                while len(futures) < workers:
                    if not submit_next():
                        break

    return aborted or crashed
