# batchr

`make` for "run this function over these 10,000 files."

`batchr` runs a Python function over a list of items (usually file paths)
in parallel, across processes. Every result is cached under a content
hash of the input file's bytes, the function's source code, and the
config you passed in. Re-running the same command skips everything
already computed. If the process is killed at item 4,000 of 10,000,
re-running resumes at 4,001 — nothing is ever recomputed unless its
inputs actually changed, and nothing is ever silently stale.

## Quickstart

### Python API

```python
from batchr import run_batch

def process(path: str) -> dict:
    with open(path) as f:
        return {"chars": len(f.read())}

report = run_batch(process, ["data/a.txt", "data/b.txt"], cache_dir=".batchr")
print(report.summary())        # "2 item(s): 2 ok, 0 cached, 0 failed, in 0.01s."
print(report.failed_items())   # []
```

Run it again — nothing recomputes:

```python
report = run_batch(process, ["data/a.txt", "data/b.txt"], cache_dir=".batchr")
print(report.summary())        # "2 item(s): 0 ok, 2 cached, 0 failed, in 0.00s."
```

### CLI

```bash
batchr run --fn mymodule:process --items "data/*.txt" --cache-dir .batchr
batchr status --cache-dir .batchr
batchr failed --cache-dir .batchr
batchr purge --older-than 30 --cache-dir .batchr
```

`--items` accepts a glob (`"data/*.npz"`), a directory (globbed
recursively), or a `.txt` file listing one item per line. `batchr run`
exits 0 if nothing failed, 1 otherwise — safe to use as a CI gate.

## The cache-key recipe

Every item's cache key is a single SHA-256 hash, updated in this exact
order:

1. **The item's bytes** — if the item is a path to an existing file, its
   content is hashed in 1 MB chunks (so multi-GB files are never read
   into memory at once); otherwise the item string itself is hashed as
   UTF-8 (for opaque, non-path keys).
2. **The function's source** — `inspect.getsource(fn)`. If source isn't
   available (a lambda from the REPL, a `functools.partial`, a
   dynamically-generated function), `batchr` falls back to
   `f"{fn.__module__}.{fn.__qualname__}"` and emits a warning — the run
   still works, but code changes to that function will no longer
   auto-invalidate its cache entries.
3. **The config** — `json.dumps(config, sort_keys=True)` (an empty dict
   if `config=None`).
4. **The serializer name** — `"pickle"`, `"json"`, or `"npz"`.

Change any one of these — the file's bytes, the function's code, or the
config — and every item whose hash depends on it gets a new cache key
and is recomputed automatically. Everything else is untouched. There is
no cache invalidation command because there's nothing to invalidate by
hand.

## Crash safety

A result only counts as cached once **both** of the following are true:
its output has been fully written to its final path, and its row exists
in the SQLite index. The write path is:

1. Serialize the result to `objects/<shard>/<key>.<ext>.tmp`.
2. `os.replace()` the tmp file onto its final path — atomic on the same
   filesystem.
3. Commit the index row (`INSERT OR REPLACE INTO cache ...`) in the
   parent process.

If the process is killed at any point before step 3 finishes, the
worst case is an orphaned tmp file or an output file with no matching
index row — never a cache hit that points at bad or partial data. On
the next run, `cache.get(key)` only returns a path when the index row
exists, so anything not fully committed is simply treated as
not-yet-computed and recomputed. Only the parent process ever writes to
SQLite; workers return plain values and the parent does all
serialization and committing.

Failed items follow the same rule: a raised exception is recorded in the
report with a full traceback and is **never** cached, so the next run
retries exactly the items that failed — nothing else.

## Cache layout on disk

```
.batchr/
├── index.sqlite            # cache_key (PK), item, output_file, created_at, fn_name
├── objects/
│   └── ab/
│       └── abcdef1234...pkl   # sharded by the first 2 hex chars of the key
└── last_report.json        # the most recent BatchReport, for `status`/`failed`
```

`index.sqlite` runs in WAL mode: safe for one writer (the parent
process) plus concurrent readers.

## When *not* to use batchr

- **I/O-bound work that wants threads, not processes.** Workers are
  `ProcessPoolExecutor` processes by design (the target workload is
  CPU-bound), so process-spawn overhead and pickling costs make batchr a
  poor fit for, say, thousands of tiny network requests — use a
  thread/async pool for that instead.
- **Non-picklable functions.** `fn` must be a module-level function
  (importable by worker processes), not a lambda or a closure. `batchr`
  detects this up front and raises `ValueError` rather than failing
  deep inside a worker.

## Dependencies

Stdlib only, plus [`tqdm`](https://github.com/tqdm/tqdm) for progress
bars (auto-disabled when stdout isn't a TTY). `numpy` is only imported
lazily, inside the `npz` serializer path — it stays fully optional
otherwise.
