from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKER_SRC = '''
def process(item):
    with open(item) as f:
        return f.read().upper()
'''

FAILING_WORKER_SRC = '''
from pathlib import Path

def process(item):
    if "bad" in Path(item).name:
        raise ValueError(f"bad item: {item}")
    with open(item) as f:
        return f.read().upper()
'''


def _run_cli(args: list[str], cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "batchr.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _env_with_pythonpath(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_cli_run_status_failed_smoke(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(5):
        (data_dir / f"f{i}.txt").write_text(f"hello {i}")

    (tmp_path / "cli_worker.py").write_text(WORKER_SRC)
    env = _env_with_pythonpath(tmp_path)
    cache_dir = tmp_path / ".batchr"

    result = _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "5 item" in result.stdout
    assert "5 ok" in result.stdout

    status = _run_cli(["status", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert status.returncode == 0
    assert "Entries: 5" in status.stdout

    failed = _run_cli(["failed", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert failed.returncode == 0
    assert "No failed items" in failed.stdout

    # second run should be all cache hits
    result2 = _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    assert result2.returncode == 0
    assert "5 cached" in result2.stdout


def test_cli_run_exit_code_1_on_failures_and_failed_command_shows_traceback(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        (data_dir / f"ok{i}.txt").write_text(f"hello {i}")
    (data_dir / "bad0.txt").write_text("boom")

    (tmp_path / "cli_worker.py").write_text(FAILING_WORKER_SRC)
    env = _env_with_pythonpath(tmp_path)
    cache_dir = tmp_path / ".batchr"

    result = _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 1
    assert "1 failed" in result.stdout

    failed = _run_cli(["failed", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert failed.returncode == 0
    assert "bad0.txt" in failed.stdout
    assert "ValueError" in failed.stdout


def test_cli_run_bad_fn_module_prints_clean_error(tmp_path):
    env = _env_with_pythonpath(tmp_path)
    (tmp_path / "data.txt").write_text("hello")

    result = _run_cli(
        ["run", "--fn", "no_such_module:process", "--items", str(tmp_path / "data.txt"),
         "--cache-dir", str(tmp_path / ".batchr")],
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr
    assert "no_such_module" in result.stderr


def test_cli_run_missing_config_file_prints_clean_error(tmp_path):
    (tmp_path / "cli_worker.py").write_text(WORKER_SRC)
    (tmp_path / "data.txt").write_text("hello")
    env = _env_with_pythonpath(tmp_path)

    result = _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(tmp_path / "data.txt"),
         "--config", str(tmp_path / "missing_config.json"),
         "--cache-dir", str(tmp_path / ".batchr")],
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_cli_purge(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "f0.txt").write_text("hello")

    (tmp_path / "cli_worker.py").write_text(WORKER_SRC)
    env = _env_with_pythonpath(tmp_path)
    cache_dir = tmp_path / ".batchr"

    _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    purge = _run_cli(["purge", "--older-than", "0", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert purge.returncode == 0
    assert "Purged 1" in purge.stdout

    status = _run_cli(["status", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert "Entries: 0" in status.stdout


def test_cli_purge_orphaned_and_status_reports_it(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "f0.txt").write_text("hello")

    (tmp_path / "cli_worker.py").write_text(WORKER_SRC)
    env = _env_with_pythonpath(tmp_path)
    cache_dir = tmp_path / ".batchr"

    _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )

    # Simulate a killed process leaving a .tmp file behind.
    objects_dir = cache_dir / "objects"
    orphan_dir = objects_dir / "zz"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "orphan.pkl.tmp").write_bytes(b"partial")

    status = _run_cli(["status", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert status.returncode == 0
    assert "Orphaned .tmp files: 1" in status.stdout

    purge = _run_cli(
        ["purge", "--older-than", "9999", "--orphaned", "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    assert purge.returncode == 0
    assert "Purged 1 orphaned .tmp file(s)." in purge.stdout
    assert not (orphan_dir / "orphan.pkl.tmp").exists()

    # the real cached entry (created above) must survive an --older-than 9999 purge
    status2 = _run_cli(["status", "--cache-dir", str(cache_dir)], cwd=tmp_path, env=env)
    assert "Entries: 1" in status2.stdout


def test_cli_run_imports_fn_module_from_invoking_cwd_without_pythonpath(tmp_path):
    """`--fn mymodule:fn` must resolve when `mymodule.py` lives in the
    invoking directory, without PYTHONPATH set.

    NOTE: this does NOT exercise the cwd-import bug that cli.py's
    os.getcwd()-into-sys.path fix addresses. Invoking via
    `python -m batchr.cli` makes Python itself prepend the cwd to sys.path
    (per `-m` semantics), so this passes regardless of whether cli.py does
    its own insertion. It only proves the `-m` invocation style keeps
    working. See
    test_cli_run_imports_fn_module_from_invoking_cwd_via_installed_entry_point
    below for the test that actually exercises the fix, via the real
    installed console-script entry point (whose sys.path does *not*
    include the cwd for free).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "f0.txt").write_text("hello")

    (tmp_path / "workmod.py").write_text(WORKER_SRC)
    cache_dir = tmp_path / ".batchr"

    result = subprocess.run(
        [sys.executable, "-m", "batchr.cli", "run", "--fn", "workmod:process",
         "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "1 ok" in result.stdout


def test_cli_run_imports_fn_module_from_invoking_cwd_via_installed_entry_point(tmp_path):
    """Actual regression test for the cwd-import bug.

    Unlike `python -m batchr.cli`, the installed `batchr` console-script
    entry point puts the script's own directory on sys.path[0], not the
    invoking cwd — so this only passes if cli.py's `_import_fn` explicitly
    inserts os.getcwd() into sys.path itself. Verified (by temporarily
    reverting that insertion in cli.py) that this test fails with
    `ModuleNotFoundError: No module named 'workmod'` on unpatched code, and
    passes once the fix is restored.
    """
    batchr_exe = shutil.which("batchr")
    if batchr_exe is None:
        pytest.skip("batchr console script not found on PATH (not installed)")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "f0.txt").write_text("hello")

    (tmp_path / "workmod.py").write_text(WORKER_SRC)
    cache_dir = tmp_path / ".batchr"

    result = subprocess.run(
        [batchr_exe, "run", "--fn", "workmod:process",
         "--items", str(data_dir), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "1 ok" in result.stdout


def test_cli_items_txt_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = []
    for i in range(3):
        p = data_dir / f"f{i}.txt"
        p.write_text(f"hello {i}")
        files.append(str(p))

    items_file = tmp_path / "items.txt"
    items_file.write_text("\n".join(files) + "\n")

    (tmp_path / "cli_worker.py").write_text(WORKER_SRC)
    env = _env_with_pythonpath(tmp_path)
    cache_dir = tmp_path / ".batchr"

    result = _run_cli(
        ["run", "--fn", "cli_worker:process", "--items", str(items_file), "--cache-dir", str(cache_dir)],
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "3 ok" in result.stdout
