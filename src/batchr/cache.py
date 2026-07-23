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
import tempfile
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
        """Write ``obj`` atomically (tmp file + rename) then commit the index row.

        The tmp filename is unique per call (not just per cache key): two
        concurrent writers computing the same key -- two independently
        invoked ``batchr run`` processes sharing a cache_dir is the normal
        way this happens -- would otherwise both write to the exact same
        deterministic ``<key>.tmp`` path, and one process's os.replace()
        would find the other's tmp file already gone (or vice versa),
        crashing with FileNotFoundError instead of the atomicity this
        docstring promises.
        """
        ext = _EXTENSIONS.get(serializer)
        if ext is None:
            raise ValueError(f"Unknown serializer: {serializer!r}")
        final_path = self._shard_path(key, ext)
        fd, tmp_name = tempfile.mkstemp(dir=final_path.parent, prefix=f".{final_path.name}-", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
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
        all_files = [f for f in self.objects_dir.rglob("*") if f.is_file()]
        tmp_files = [f for f in all_files if f.suffix == ".tmp"]
        size_bytes = sum(f.stat().st_size for f in all_files)
        return {
            "entries": count,
            "size_bytes": size_bytes,
            "cache_dir": str(self.cache_dir),
            "orphaned_tmp_files": len(tmp_files),
            "orphaned_tmp_bytes": sum(f.stat().st_size for f in tmp_files),
        }

    def purge_orphaned_tmp_files(self) -> int:
        """Delete leftover ``*.tmp`` files under ``objects/``. Returns count removed.

        A ``.tmp`` file is written by ``put()`` and immediately renamed onto
        its final path (see module docstring / README "Crash safety"); the
        only way one is left lying around is a process getting killed
        between the write and the rename. It never has a matching index
        row (the row is only committed after the rename), so it can't be a
        cache hit — it's pure disk waste. Only safe to call when no other
        ``batchr`` process is currently writing to this cache_dir, since a
        write in progress also has a ``.tmp`` file on disk momentarily.
        """
        removed = 0
        for path in self.objects_dir.rglob("*.tmp"):
            path.unlink()
            removed += 1
        return removed

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
                    shard_dir = path.parent
                    if shard_dir != self.objects_dir and not any(shard_dir.iterdir()):
                        shard_dir.rmdir()
            conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()
        return len(rows)
