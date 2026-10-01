"""make run -> make stop against temp dirs, so the real .run/, logs/ and data/ are never touched."""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

import pytest

from pipeline.config import PROJECT_ROOT
from tests.conftest import clean_env

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("make") is None, reason="make not installed"),
    pytest.mark.skipif(not (PROJECT_ROOT / ".venv" / "bin" / "python").exists(),
                       reason="run `make install` first"),
]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


PIPELINE = ("simulator", "preprocess", "train", "infer")
NAMES = PIPELINE + ("dashboard",)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


@pytest.fixture
def dirs(tmp_path):
    run_dir, data_dir = tmp_path / "run", tmp_path / "data"
    env = clean_env(DASHBITE_RUN_DIR=str(run_dir), DASHBITE_DATA_DIR=str(data_dir),
                    SIM_INTERVAL_SECONDS="0.2", DASHBOARD_PORT=str(_free_port()))
    yield run_dir, data_dir, env
    # Never leave a stage running if an assertion failed midway.
    for name in NAMES:
        pid_path = run_dir / ".run" / f"{name}.pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


def _make(target, env, cwd):
    return subprocess.run(
        ["make", "-f", str(PROJECT_ROOT / "Makefile"), target],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
    )


def test_run_then_stop(dirs, tmp_path):
    run_dir, data_dir, env = dirs
    # One knob sets every poller's cadence, as in class (where it's 15s).
    env = {**env, "POLL_INTERVAL_SECONDS": "0.2", "TRAIN_EVERY_N_EVENTS": "50"}
    pid_paths = {n: run_dir / ".run" / f"{n}.pid" for n in NAMES}
    log_paths = {n: run_dir / "logs" / f"{n}.log" for n in NAMES}

    first = _make("run", env, tmp_path)
    assert first.returncode == 0, first.stderr
    pids = {n: int(pid_paths[n].read_text()) for n in NAMES}
    for name, pid in pids.items():
        assert f"started {name} (pid {pid})" in first.stdout
        assert _alive(pid)
        # Its own session leader: closing the launching terminal can't take it down.
        assert os.getsid(pid) == pid
    assert f"open it:  http://localhost:{env['DASHBOARD_PORT']}" in first.stdout

    deadline = time.monotonic() + 30
    while True:
        try:
            health = urllib.request.urlopen(
                f"http://localhost:{env['DASHBOARD_PORT']}/_stcore/health", timeout=1).read()
        except OSError:
            health = b""
        if health == b"ok":
            break
        assert time.monotonic() < deadline, "dashboard not healthy within 30s"
        time.sleep(0.25)

    # The stages hand off through data/ with no other command, all the way to predictions.
    deadline = time.monotonic() + 25
    while not list((data_dir / "predictions").glob("predictions_*.csv")):
        assert time.monotonic() < deadline, "no predictions appeared"
        time.sleep(0.1)
    assert (data_dir / "models" / "model_v0001.json").exists()

    second = _make("run", env, tmp_path)
    assert second.returncode == 0, second.stderr
    for name, pid in pids.items():
        assert f"{name} already running (pid {pid})" in second.stdout
        assert int(pid_paths[name].read_text()) == pid

    started = time.monotonic()
    stop = _make("stop", env, tmp_path)
    assert stop.returncode == 0, stop.stderr
    assert time.monotonic() - started < 10
    for name, pid in pids.items():
        assert f"stopped {name} (pid {pid})" in stop.stdout
        assert not pid_paths[name].exists()
        assert not _alive(pid)
    for name in PIPELINE:
        assert "stopped after" in log_paths[name].read_text().strip().splitlines()[-1]
    assert "Stopping..." in log_paths["dashboard"].read_text()

    again = _make("stop", env, tmp_path)
    assert again.returncode == 0, again.stderr
    assert again.stdout.strip() == "nothing running"


def test_run_with_bad_config_starts_nothing(dirs, tmp_path):
    run_dir, _, env = dirs
    result = _make("run", {**env, "SIM_INTERVAL_SECONDS": "0"}, tmp_path)
    assert result.returncode != 0
    assert "SIM_INTERVAL_SECONDS" in result.stderr
    assert not list(run_dir.glob(".run/*.pid"))


def _hold_port(port):
    """Occupy ``port`` the way another Streamlit (or anything) would."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", port))
    sock.listen()
    return sock


def _make_merged(target, env, cwd):
    """stdout and stderr through one pipe, so the test sees the order a user would."""
    return subprocess.run(["make", "-f", str(PROJECT_ROOT / "Makefile"), target], cwd=cwd,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=30)


def _running(env, tmp_path):
    out = _make("status", env, tmp_path).stdout.splitlines()
    return {line.split()[0] for line in out if " running (pid " in line}


def test_busy_dashboard_port_fails_only_the_dashboard_then_run_recovers(dirs, tmp_path):
    run_dir, _, env = dirs
    port = int(env["DASHBOARD_PORT"])
    holder = _hold_port(port)
    try:
        result = _make_merged("run", env, tmp_path)
    finally:
        holder.close()

    assert result.returncode != 0  # make run says something went wrong
    lines = result.stdout.splitlines()
    failure = next(i for i, l in enumerate(lines) if l.startswith("dashboard exited right away"))
    assert f"port {port} is already in use" in lines[failure]  # the log's last line, inline
    # Regression: with output piped, the failure used to print before the
    # "started …" lines it followed, because stdout was block-buffered.
    started = [i for i, l in enumerate(lines) if l.startswith("started ")]
    assert len(started) == 4 and max(started) < failure
    assert _running(env, tmp_path) == set(PIPELINE)  # the other four keep running
    assert not (run_dir / ".run" / "dashboard.pid").exists()

    # Port free again: `make run` starts only what's missing.
    again = _make("run", env, tmp_path)
    assert again.returncode == 0, again.stderr
    assert again.stdout.count("already running") == 4
    assert "started dashboard" in again.stdout
    assert _running(env, tmp_path) == set(NAMES)
    assert _make("stop", env, tmp_path).returncode == 0


def test_crashed_stage_shows_as_stopped_and_run_restarts_only_it(dirs, tmp_path):
    run_dir, _, env = dirs
    assert _make("run", env, tmp_path).returncode == 0
    pids = {n: int((run_dir / ".run" / f"{n}.pid").read_text()) for n in NAMES}

    os.kill(pids["infer"], signal.SIGKILL)  # a crash: no clean shutdown, PID file left behind
    deadline = time.monotonic() + 5
    while _alive(pids["infer"]):
        assert time.monotonic() < deadline
        time.sleep(0.05)

    status = _make("status", env, tmp_path).stdout
    assert "infer       stopped" in status and status.rstrip().endswith("4/5 running")

    again = _make("run", env, tmp_path)
    assert again.returncode == 0, again.stderr
    assert "started infer" in again.stdout and again.stdout.count("already running") == 4
    new_infer = int((run_dir / ".run" / "infer.pid").read_text())
    assert new_infer != pids["infer"] and _alive(new_infer)
    for name in set(NAMES) - {"infer"}:  # nobody else was restarted
        assert int((run_dir / ".run" / f"{name}.pid").read_text()) == pids[name]

    stop = _make("stop", env, tmp_path)
    assert stop.stdout.count("stopped ") == 5


def test_make_logs_before_anything_ran(dirs, tmp_path):
    result = _make("logs", dirs[2], tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "no logs yet: start the stack with make run"
