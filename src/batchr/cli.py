"""The `batchr` command-line interface."""

from __future__ import annotations

import argparse
import glob as glob_module
import importlib
import json
import sys
from pathlib import Path

from .cache import CacheStore
from .core import load_last_report, run_batch


def _import_fn(spec: str):
    if ":" not in spec:
        raise ValueError(f"--fn must be in 'module:function' form, got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as e:
        raise ValueError(f"Module {module_name!r} has no attribute {func_name!r}") from e


def _resolve_items(spec: str) -> list[str]:
    path = Path(spec)
    if path.is_dir():
        return sorted(str(p) for p in path.rglob("*") if p.is_file())
    if path.is_file() and path.suffix == ".txt":
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(glob_module.glob(spec, recursive=True))


def _cmd_run(args: argparse.Namespace) -> int:
    fn = _import_fn(args.fn)
    items = _resolve_items(args.items)
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    report = run_batch(
        fn,
        items,
        cache_dir=args.cache_dir,
        config=config,
        workers=args.workers,
        retries=args.retries,
        force=args.force,
        fail_fast=args.fail_fast,
        serializer=args.serializer,
    )
    print(report.summary())
    return 0 if report.failed == 0 else 1


def _cmd_status(args: argparse.Namespace) -> int:
    store = CacheStore(Path(args.cache_dir))
    stats = store.stats()
    report = load_last_report(args.cache_dir)
    print(f"Cache dir: {stats['cache_dir']}")
    print(f"Entries: {stats['entries']}")
    print(f"Size on disk: {stats['size_bytes']} bytes")
    print(f"Last run: {report.summary() if report else 'no report found'}")
    return 0


def _cmd_failed(args: argparse.Namespace) -> int:
    report = load_last_report(args.cache_dir)
    if report is None:
        print("No last report found.")
        return 0
    failed = [r for r in report.results if r.status == "failed"]
    if not failed:
        print("No failed items in the last run.")
        return 0
    for r in failed:
        print(f"=== {r.item} ===")
        print(r.error)
    return 0


def _cmd_purge(args: argparse.Namespace) -> int:
    store = CacheStore(Path(args.cache_dir))
    n = store.purge(args.older_than)
    print(f"Purged {n} entries older than {args.older_than} day(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batchr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run fn over items, using the cache")
    p_run.add_argument("--fn", required=True, help="module:function")
    p_run.add_argument("--items", required=True, help="glob, directory, or .txt file of items")
    p_run.add_argument("--config", default=None, help="path to a JSON config file, hashed into the cache key")
    p_run.add_argument("--workers", type=int, default=0)
    p_run.add_argument("--retries", type=int, default=0)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--fail-fast", action="store_true", dest="fail_fast")
    p_run.add_argument("--cache-dir", default=".batchr")
    p_run.add_argument("--serializer", default="pickle", choices=["pickle", "json", "npz"])
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="Show cache counts, size, and last run")
    p_status.add_argument("--cache-dir", default=".batchr")
    p_status.set_defaults(func=_cmd_status)

    p_failed = sub.add_parser("failed", help="List items that failed in the last run, with tracebacks")
    p_failed.add_argument("--cache-dir", default=".batchr")
    p_failed.set_defaults(func=_cmd_failed)

    p_purge = sub.add_parser("purge", help="Delete cache entries older than N days")
    p_purge.add_argument("--older-than", type=float, default=30)
    p_purge.add_argument("--cache-dir", default=".batchr")
    p_purge.set_defaults(func=_cmd_purge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
