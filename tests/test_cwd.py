"""Regression tests: the cache must survive cwd changes and relocation.

Notebook sessions (Colab/Kaggle) chdir freely between cells, and persistent
storage gets re-mounted at different absolute paths. Neither may cause cache
misses or stale lookups.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from batchr import run_batch
from batchr.cache import CacheStore

from .workers import read_upper


@pytest.fixture
def restore_cwd():
    old = os.getcwd()
    yield
    os.chdir(old)


def test_relative_cache_dir_pinned_at_construction(tmp_path, restore_cwd):
    workdir = tmp_path / "work"
    workdir.mkdir()
    os.chdir(workdir)
    store = CacheStore(".batchr")
    assert store.cache_dir == (workdir / ".batchr").resolve()


def test_cache_hits_survive_cwd_change(tmp_path, restore_cwd):
    workdir = tmp_path / "work"
    elsewhere = tmp_path / "elsewhere"
    workdir.mkdir()
    elsewhere.mkdir()
    data = workdir / "a.txt"
    data.write_text("hello")

    os.chdir(workdir)
    cache_dir = workdir / ".batchr"
    report = run_batch(read_upper, [str(data)], cache_dir=cache_dir)
    assert report.ok == 1

    os.chdir(elsewhere)
    report = run_batch(read_upper, [str(data)], cache_dir=cache_dir)
    assert report.cached == 1
    assert report.ok == 0


def test_get_survives_cwd_change_with_relative_cache_dir(tmp_path, restore_cwd):
    workdir = tmp_path / "work"
    elsewhere = tmp_path / "elsewhere"
    workdir.mkdir()
    elsewhere.mkdir()

    os.chdir(workdir)
    store = CacheStore(".batchr")
    store.put("k", "item", "value", "pickle")

    os.chdir(elsewhere)
    path = store.get("k")
    assert path is not None
    assert path.exists()


def test_cache_dir_is_relocatable(tmp_path):
    old_dir = tmp_path / "old" / ".batchr"
    store = CacheStore(old_dir)
    store.put("k", "item", "value", "pickle")

    new_dir = tmp_path / "new" / ".batchr"
    new_dir.parent.mkdir()
    shutil.move(str(old_dir), str(new_dir))

    moved = CacheStore(new_dir)
    path = moved.get("k")
    assert path is not None
    assert path.exists()
    assert str(path).startswith(str(new_dir.resolve()))


def test_index_rows_are_relative(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("k", "item", "value", "pickle")
    conn = sqlite3.connect(store.db_path)
    (row,) = conn.execute("SELECT output_file FROM cache").fetchone()
    conn.close()
    assert not Path(row).is_absolute()


def test_absolute_rows_from_old_caches_still_resolve(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    final = store.put("k", "item", "value", "pickle")
    # rewrite the row the way pre-fix versions stored it: absolute
    conn = sqlite3.connect(store.db_path)
    conn.execute("UPDATE cache SET output_file = ? WHERE cache_key = 'k'", (str(final),))
    conn.commit()
    conn.close()
    assert store.get("k") == final


def test_purge_deletes_files_from_relative_rows(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    final = store.put("k", "item", "value", "pickle")
    conn = sqlite3.connect(store.db_path)
    conn.execute("UPDATE cache SET created_at = ?", (time.time() - 40 * 86400,))
    conn.commit()
    conn.close()
    assert store.purge(older_than_days=30) == 1
    assert not final.exists()
