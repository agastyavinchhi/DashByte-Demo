from __future__ import annotations

import subprocess
import sys

import pytest

from pipeline import runner

pytestmark = pytest.mark.unit


def _write_pid(run_dir, name, content):
    path = runner.pid_file(run_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_no_pid_file_means_not_running(tmp_path):
    assert runner.running_pid(tmp_path, "simulator") is None


def test_live_pid_means_already_running(tmp_path, monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr(runner, "_is_ours", lambda pid, name: True)
        path = _write_pid(tmp_path, "simulator", f"{proc.pid}\n")
        assert runner.running_pid(tmp_path, "simulator") == proc.pid
        assert path.exists()
    finally:
        proc.kill()
        proc.wait()


def test_dead_pid_is_stale_and_removed(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    path = _write_pid(tmp_path, "simulator", f"{proc.pid}\n")
    assert runner.running_pid(tmp_path, "simulator") is None
    assert not path.exists()


def test_reused_pid_of_another_program_is_stale(tmp_path):
    # The PID is alive but it isn't the simulator, so stop must never signal it.
    proc = subprocess.Popen(["sleep", "30"])
    try:
        path = _write_pid(tmp_path, "simulator", f"{proc.pid}\n")
        assert runner.running_pid(tmp_path, "simulator") is None
        assert not path.exists()
    finally:
        proc.kill()
        proc.wait()


def test_garbage_pid_file_is_stale_and_removed(tmp_path):
    path = _write_pid(tmp_path, "simulator", "not a pid")
    assert runner.running_pid(tmp_path, "simulator") is None
    assert not path.exists()


def test_stop_with_nothing_running(tmp_path, capsys):
    assert runner.stop(tmp_path) == 0
    assert capsys.readouterr().out.strip() == "nothing running"


def test_paths_live_under_run_dir(tmp_path):
    assert runner.pid_file(tmp_path, "simulator") == tmp_path / ".run" / "simulator.pid"
    assert runner.log_file(tmp_path, "simulator") == tmp_path / "logs" / "simulator.log"


def test_status_lists_every_process(tmp_path, capsys):
    assert runner.status(tmp_path) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in out[:-1]] == list(runner.PROCESSES)
    assert all(line.split()[1] == "stopped" for line in out[:-1])
    assert out[-1] == f"0/{len(runner.PROCESSES)} running"


def test_stop_goes_in_reverse_start_order(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "running_pid", lambda run_dir, name: 1)
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(runner, "_wait_for_exit", lambda pid, timeout: True)
    runner.stop(tmp_path)
    stopped = [line.split()[1] for line in capsys.readouterr().out.splitlines()]
    assert stopped == list(reversed(list(runner.PROCESSES)))
