from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from batchr.cache import CacheStore


def test_put_get_roundtrip_pickle(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    path = store.put("deadbeef", "item1", {"a": 1}, "pickle")
    assert path.exists()
    assert store.get("deadbeef") == path

    import pickle

    with open(path, "rb") as f:
        assert pickle.load(f) == {"a": 1}


def test_put_get_roundtrip_json(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    path = store.put("cafebabe", "item1", {"a": 1}, "json")
    assert store.get("cafebabe") == path

    import json

    with open(path) as f:
        assert json.load(f) == {"a": 1}


def test_get_missing_key_returns_none(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    assert store.get("does-not-exist") is None


def test_sharding_by_first_two_hex_chars(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    key = "ab1234567890"
    path = store.put(key, "item1", "value", "pickle")
    assert path.parent.name == "ab"
    assert path.parent.parent.name == "objects"


def test_no_tmp_files_remain_after_put(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    for i in range(5):
        store.put(f"key{i}", f"item{i}", i, "pickle")

    tmp_files = list((tmp_path / ".batchr" / "objects").rglob("*.tmp"))
    assert tmp_files == []


def test_stats_counts_and_size(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("key1", "item1", "hello world", "pickle")
    store.put("key2", "item2", "another value", "pickle")

    stats = store.stats()
    assert stats["entries"] == 2
    assert stats["size_bytes"] > 0
    assert stats["cache_dir"] == str(tmp_path / ".batchr")


def test_purge_removes_old_entries(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("old_key", "item_old", "old", "pickle")
    store.put("new_key", "item_new", "new", "pickle")

    # backdate the "old_key" row directly, since created_at defaults to now()
    conn = sqlite3.connect(store.db_path)
    old_time = time.time() - (40 * 86400)
    conn.execute("UPDATE cache SET created_at = ? WHERE cache_key = ?", (old_time, "old_key"))
    conn.commit()
    conn.close()

    removed = store.purge(older_than_days=30)
    assert removed == 1
    assert store.get("old_key") is None
    assert store.get("new_key") is not None
    assert store.stats()["entries"] == 1

    # The now-empty "ol" shard dir should be cleaned up, not left as litter;
    # "ne" (still holding new_key's file) must be untouched.
    assert not (store.objects_dir / "ol").exists()
    assert (store.objects_dir / "ne").exists()


def test_put_overwrites_existing_key(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("k", "item", "first", "pickle")
    path = store.put("k", "item", "second", "pickle")

    import pickle

    with open(path, "rb") as f:
        assert pickle.load(f) == "second"
    assert store.stats()["entries"] == 1


def test_concurrent_put_same_key_does_not_crash(tmp_path):
    # Two writers computing the same cache key -- the normal way this
    # happens is two independently invoked `batchr run` processes sharing
    # a cache_dir -- used to both write to the exact same deterministic
    # "<key>.tmp" path. One's os.replace() would then find the other's
    # tmp file already gone (or vice versa), crashing with
    # FileNotFoundError instead of completing atomically.
    import threading

    store = CacheStore(tmp_path / ".batchr")
    errors = []

    def writer(value):
        try:
            store.put("samekey", "item", {"v": value}, "pickle")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.stats()["entries"] == 1
    assert store.stats()["orphaned_tmp_files"] == 0


def test_get_returns_none_if_output_file_missing(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    path = store.put("k", "item", "value", "pickle")
    path.unlink()
    assert store.get("k") is None


def test_purge_orphaned_tmp_files_removes_leftover_tmp_only(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("key1", "item1", "hello", "pickle")

    # Simulate a process killed between _serialize() and os.replace() in put().
    orphan_dir = store.objects_dir / "de"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / "deadbeef.pkl.tmp"
    orphan.write_bytes(b"partial")

    removed = store.purge_orphaned_tmp_files()

    assert removed == 1
    assert not orphan.exists()
    # the real, committed entry must be untouched
    assert store.get("key1") is not None


def test_purge_orphaned_tmp_files_no_op_when_none_exist(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("key1", "item1", "hello", "pickle")
    assert store.purge_orphaned_tmp_files() == 0


def test_stats_reports_orphaned_tmp_files(tmp_path):
    store = CacheStore(tmp_path / ".batchr")
    store.put("key1", "item1", "hello", "pickle")

    orphan_dir = store.objects_dir / "de"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "deadbeef.pkl.tmp").write_bytes(b"partial data")

    stats = store.stats()
    assert stats["orphaned_tmp_files"] == 1
    assert stats["orphaned_tmp_bytes"] == len(b"partial data")
