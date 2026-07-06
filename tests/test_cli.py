from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
