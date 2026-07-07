"""On-disk cache store: a SQLite index plus sharded result files.

Layout inside ``cache_dir``::

    .batchr/
    ├── index.sqlite          # table: cache_key PK, item, output_file, created_at, fn_name
    └── objects/ab/abcdef...  # results sharded by first 2 hex chars of the key

Only the parent process ever writes to the SQLite index. Worker processes
return plain values; the parent serializes them and commits the index row.
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

_EXTENSIONS = {"pickle": ".pkl", "json": ".json", "npz": ".npz"}


def _serialize(obj: Any, path: Path, serializer: str) -> None:
    if serializer == "pickle":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif serializer == "json":
        with open(path, "w") as f:
            json.dump(obj, f)
    elif serializer == "npz":
        import numpy as np  # lazy import: numpy stays an optional dependency

        with open(path, "wb") as f:
            np.savez(f, **obj)
    else:
        raise ValueError(f"Unknown serializer: {serializer!r}")


def load(path: Path, serializer: str) -> Any:
    """Load a previously cached result back into memory."""
    if serializer == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif serializer == "json":
        with open(path) as f:
            return json.load(f)
    elif serializer == "npz":
        import numpy as np

        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    else:
        raise ValueError(f"Unknown serializer: {serializer!r}")


class CacheStore:
    """The cache for a single ``cache_dir``."""

    def __init__(self, cache_dir: str | Path):
        # Pin a relative cache_dir to the cwd at construction time. Notebook
        # and CLI sessions chdir freely between runs; without this, the same
        # ".batchr" silently becomes a different directory.
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.objects_dir = self.cache_dir / "objects"
        self.db_path = self.cache_dir / "index.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    item TEXT NOT NULL,
                    output_file TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    fn_name TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _shard_path(self, key: str, ext: str) -> Path:
        shard_dir = self.objects_dir / key[:2]
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{key}{ext}"

    def get(self, key: str) -> Path | None:
        """Return the output path for ``key`` if it is fully committed, else None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT output_file FROM cache WHERE cache_key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        path = Path(row[0])
        # Rows are stored relative to cache_dir so the cache directory can be
        # moved or re-mounted (Drive, /kaggle/working) without invalidating
        # anything. Absolute rows from caches written by older versions still
        # resolve as-is.
        if not path.is_absolute():
            path = self.cache_dir / path
        if not path.exists():
            return None
        return path

    def put(self, key: str, item: str, obj: Any, serializer: str, fn_name: str = "") -> Path:
        """Write ``obj`` atomically (tmp file + rename) then commit the index row."""
        ext = _EXTENSIONS.get(serializer)
        if ext is None:
            raise ValueError(f"Unknown serializer: {serializer!r}")
        final_path = self._shard_path(key, ext)
        tmp_path = final_path.with_name(final_path.name + ".tmp")
        _serialize(obj, tmp_path, serializer)
        os.replace(tmp_path, final_path)

        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(cache_key, item, output_file, created_at, fn_name) VALUES (?, ?, ?, ?, ?)",
                (key, item, str(final_path.relative_to(self.cache_dir)), time.time(), fn_name),
            )
            conn.commit()
        finally:
            conn.close()
        return final_path

    def stats(self) -> dict:
        conn = self._connect()
        try:
            (count,) = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        finally:
            conn.close()
        size_bytes = sum(f.stat().st_size for f in self.objects_dir.rglob("*") if f.is_file())
        return {
            "entries": count,
            "size_bytes": size_bytes,
            "cache_dir": str(self.cache_dir),
        }

    def purge(self, older_than_days: float) -> int:
        """Delete entries (and their output files) older than ``older_than_days``. Returns count removed."""
        cutoff = time.time() - older_than_days * 86400
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT cache_key, output_file FROM cache WHERE created_at < ?", (cutoff,)
            ).fetchall()
            for _key, output_file in rows:
                path = Path(output_file)
                if not path.is_absolute():
                    path = self.cache_dir / path
                if path.exists():
                    path.unlink()
            conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()
        return len(rows)
